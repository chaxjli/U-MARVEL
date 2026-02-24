from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

# 导入相关模型组件和工具类
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F

# 导入自定义的双向注意力模型
from models.qwen2_vl_bi_model import Qwen2VLBiModel

class Similarity(nn.Module):
    """相似度计算模块，支持余弦相似度计算
    通过温度参数控制相似度分布的锐化程度
    
    参数：
        temp (float): 温度系数，默认0.07
    """
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp  # 温度参数控制相似度分布的锐化程度
        self.cos = nn.CosineSimilarity(dim=-1)  # 余弦相似度计算器

    def forward(self, x, y):
        """计算两个张量的余弦相似度并缩放
        输入形状：[batch_size, embedding_dim]
        输出形状：[batch_size] 或 [batch_size, batch_size]（矩阵形式）
        """
        return self.cos(x, y) / self.temp

class Qwen2VLRetFinetuneForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """支持对比学习微调的多模态生成模型
    核心功能扩展：
    - 支持双向注意力机制切换
    - 实现特征提取接口
    - 对比损失计算（InfoNCE）
    - 分布式训练支持
    - 多模态输入处理（文本/图像/视频）
    """
    def __init__(self, config):
        """初始化方法扩展：
        - 根据配置动态选择基础模型架构
        - 添加池化方式配置
        """
        super().__init__(config)
        # 动态切换双向注意力模型
        if getattr(config, "use_bi_attn", False):
            self.model = Qwen2VLBiModel(config)
        # 配置特征池化方式（均值池化或特定标记池化）
        self.mean_pooling = getattr(config, "mean_pooling", False)

    def get_features(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        """特征提取核心方法
        处理流程：
        1. 处理多模态嵌入（文本/图像/视频）
        2. 通过模型前向传播
        3. 根据配置策略提取特征
        """
        # 设置输出控制参数
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        # 多模态嵌入处理逻辑
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # 处理图像嵌入
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())  # 确保数据类型一致
                image_embeds = self.visual(
                    pixel_values, 
                    grid_thw=image_grid_thw  # 图像网格参数（tile_height_width）
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id  # 定位图像标记位置
                if self.training:
                    inputs_embeds = inputs_embeds.clone()  # 训练时克隆张量保持计算图
                inputs_embeds[image_mask] = image_embeds  # 替换图像标记位置的嵌入
            
            # 处理视频嵌入（逻辑与图像类似）
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(
                    pixel_values_videos,
                    grid_thw=video_grid_thw
                ).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            
            # 设备对齐
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 模型前向传播
        outputs = self.model(
            input_ids=None,  # 使用预计算的inputs_embeds
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 获取最终隐藏状态 [batch_size, seq_len, hidden_dim]
        
        # 特征提取策略
        embed_index = self.config.emb_token_ids[0]  # 目标特征标记ID
        if self.mean_pooling:
            return self.extract_mean_embed_features(hidden_states, labels, embed_index)
        else:
            # 定位目标标记位置
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
            # 提取对应位置的隐藏状态
            return hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None 
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        """前向传播流程扩展：
        1. 分批次处理大输入（mini-batch）
        2. 多模态嵌入处理
        3. 特征提取与对比损失计算
        4. 分布式训练支持
        """
        # 配置输出控制
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        # 分批次处理逻辑（解决大batch内存问题）
        mini_batch_size = 32 
        input_ids_list = torch.split(input_ids, mini_batch_size)
        attention_mask_list = torch.split(attention_mask, mini_batch_size)
        
        # 图像处理相关参数初始化
        if image_grid_thw is not None:
            # 计算图像块的累积和用于分批次提取
            cumsum_pixel_values = torch.cumsum(image_grid_thw[:, 1] * image_grid_thw[:, 2], dim=-1)
            cumsum_pixel_values = torch.cat((
                torch.tensor([0], device=cumsum_pixel_values.device),
                cumsum_pixel_values
            ))
            image_nums = 0  # 已处理图像计数器
        
        all_hidden_states = []  # 存储各批次隐藏状态

        # 分批次处理循环
        for i in range(len(input_ids_list)):
            # 当前批次处理
            batch_inputs_embeds = None
            if inputs_embeds is None:
                batch_inputs_embeds = self.model.embed_tokens(input_ids_list[i])
                
                # 处理当前批次的图像嵌入
                if pixel_values is not None:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    current_image_num = torch.sum(torch.any(image_mask, dim=-1)).item()
                    
                    if current_image_num != 0:
                        # 提取对应图像块
                        batch_pixel_values = pixel_values[
                            cumsum_pixel_values[image_nums] : cumsum_pixel_values[image_nums + current_image_num]
                        ]
                        batch_pixel_values = batch_pixel_values.type(self.visual.get_dtype())
                        
                        # 生成图像嵌入
                        batch_image_embeds = self.visual(
                            batch_pixel_values,
                            grid_thw=image_grid_thw[image_nums:image_nums + current_image_num]
                        ).to(batch_inputs_embeds.device)
                        
                        image_nums += current_image_num  # 更新计数器
                        
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        
                        # 替换图像标记位置的嵌入
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                
                # 处理视频嵌入（逻辑与图像类似）
                if pixel_values_videos is not None:
                    # ...（类似图像处理逻辑）
                
                # 设备对齐
                if attention_mask is not None:
                    batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)

            # 模型前向传播
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=batch_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=batch_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]  # 当前批次隐藏状态
            all_hidden_states.append(hidden_states)

        # 合并所有批次的隐藏状态
        hidden_states = torch.cat(all_hidden_states)

        # 确定有效batch大小（考虑负样本）
        if has_hard_negative:
            batch_size = len(hidden_states) // 3  # 正例+负例+困难负例
        elif not inference:
            batch_size = len(hidden_states) // 2  # 正例+负例
        else:
            batch_size = len(hidden_states)  # 推理模式

        # 特征提取
        embed_index = self.config.emb_token_ids[0]
        if self.mean_pooling:
            embed_features = self.extract_mean_embed_features(hidden_states, labels, embed_index)
        else:
            # 定位目标标记位置
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1)
            # 提取对应位置前一时刻的特征（因果语言模型特性）
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

        # 推理模式直接返回特征
        if inference:
            if ids is not None:
                return embed_features, ids  # 带ID返回
            elif qids or dids:
                return embed_features, qids, dids  # 查询/文档ID
            return embed_features

        # 划分正负样本特征
        if has_hard_negative:
            embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
        else:
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]

        # 分布式训练同步特征
        if dist.is_initialized():
            # 同步困难负样本
            if has_hard_negative:
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(embed3_list, embed3.contiguous())
                embed3_list[dist.get_rank()] = embed3
                embed3 = torch.cat(embed3_list, 0)
            
            # 同步正负样本
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            dist.all_gather(embed1_list, embed1.contiguous())
            dist.all_gather(embed2_list, embed2.contiguous())
            
            # 保持当前进程的原始梯度
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        # 计算对比损失
        sim = Similarity(temp=0.05)  # 相似度计算器
        embed1 = F.normalize(embed1, dim=-1)  # L2归一化
        embed2 = F.normalize(embed2, dim=-1)
        
        # 构建相似度矩阵
        cos_sim = sim(embed1.unsqueeze(1), embed2.unsqueeze(0))  # [batch_size, batch_size]
        
        # 合并困难负样本
        if has_hard_negative:
            embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)  # 列维度拼接
        
        # 构造对比学习标签（对角线为匹配对）
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, nce_labels)
        
        return SequenceClassifierOutput(loss=loss)

    def extract_mean_embed_features(self, hidden_states, labels, embed_index):
        """均值池化特征提取方法
        处理逻辑：
        1. 根据标签中的-100确定有效序列起始位置
        2. 根据目标标记确定有效序列结束位置
        3. 截取中间特征进行均值池化
        
        参数:
            hidden_states: [batch_size, seq_len, dim]
            labels: [batch_size, seq_len]
            embed_index: 目标标记ID
        
        返回:
            [batch_size, dim]
        """
        bs, seq, dim = hidden_states.shape
        embed_features = torch.zeros(bs, dim, device=hidden_states.device)

        for i in range(bs):
            # 确定有效序列范围
            mask_neg_100 = labels[i] == -100
            start_pos = torch.where(mask_neg_100)[0].max().item() + 1 if mask_neg_100.any() else 0
            
            mask_embed = labels[i] == embed_index
            end_pos = torch.where(mask_embed)[0].min().item() if mask_embed.any() else seq

            # 特征截取与池化
            if start_pos < end_pos:
                features = hidden_states[i, start_pos:end_pos]
                embed_features[i] = features.mean(dim=0)
            else:
                # 处理无效范围情况
                print("Warning: Invalid feature range")
                embed_features[i] = torch.zeros(dim, device=hidden_states.device)

        return embed_features