from typing import Tuple, Optional, List, Union 
import torch
# 适配 npu
import torch_npu
from torch_npu.contrib import transfer_to_npu
# 从 transformers 库的 utils 模块导入日志记录工具
# 获取当前模块的日志记录器
from transformers.utils import logging
logger = logging.get_logger(__name__)

# 从 transformers 库导入多个类，用于自动加载处理器、模型等
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from torch import nn 
import torch.distributed as dist

# 从 transformers 库的 modeling_outputs 模块导入 SequenceClassifierOutput 类，用于分类任务的输出
from transformers.modeling_outputs import SequenceClassifierOutput

# 从 transformers 库的 qwen2_vl 模块导入 Qwen2VLCausalLMOutputWithPast 类，用于 Qwen2-VL 模型的因果语言模型输出
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F

# 定义一个相似度计算类，继承自 nn.Module
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07):
        # 调用父类的构造函数
        super().__init__()
        # 初始化温度参数
        self.temp = temp
        # 初始化余弦相似度计算模块
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        # 计算余弦相似度并除以温度参数
        return self.cos(x, y) / self.temp

# 定义一个继承自 Qwen2VLForConditionalGeneration 的类，用于条件生成和检索任务
class Qwen2VLRetForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def forward(
        self,
        # 输入的 token ID 张量； 注意力掩码张量，可选 ；位置 ID 张量，可选 ；past的键值对列表，可选
        # 标签张量，可选  # 是否使用缓存，可选； # 是否输出注意力权重，可选
        # 是否输出隐藏状态，可选               # 是否返回字典形式的输出，可选                      # 图像像素值张量，可选 # 视频像素值张量，可选
        # 图像网格的时间、高度、宽度张量，可选    # 视频网格的时间、高度、宽度张量，可选     
        # 是否为推理模式，默认为 False          # 是否有难负样本，默认为 False           # 查询 ID 列表，可选  # 文档 ID 列表，可选
        # 通用 ID 列表，可选 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        
        # past_key_values 中保存的就是之前生成词对应的键和值对，模型会将当前输入词的查询与 past_key_values 结合起来，
        # 计算出对生成下一个词有帮助的注意力分数，以生成下一个词的表示
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入的嵌入张量，可选；  
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,  # RoPE 偏移量张量，可选
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None 
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            标签（形状为 (batch_size, sequence_length) 的 torch.LongTensor，可选）：
            用于计算掩码语言模型损失的标签。索引值应要么在 [0, ..., config.vocab_size] 范围内，
            要么为 -100（请参阅 input_ids 的文档字符串）。索引设置为 -100 的标记将被忽略（掩码处理），
            损失仅针对标签在 [0, ..., config.vocab_size] 范围内的标记进行计算。
        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        "这幅图展示了一个街道场景，前景中有一个红色的停车标志。背景里有一扇带有汉字的大红门..."
        ```"""
        # 如果未指定 output_attentions，则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 如果输入嵌入张量未提供，则通过词嵌入层将输入的 token ID 转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # 处理图片，将 image_token 替换为真正的图片
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype()) # 将图像像素值的类型转换为视觉模块所需的类型
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device) # 将图像转换为嵌入向量
                # 假设 input_ids 是 [5, 3, 10, 3]，而 image_token_id 是 3 张量操作
                # image_mask 就会是 [False, True, False, True] 张量
                image_mask = input_ids == self.config.image_token_id # 找出输入中图像 token 的位置 
                if self.training:
                    inputs_embeds = inputs_embeds.clone() # 在训练模式下，克隆输入嵌入张量，避免原地修改 why
                inputs_embeds[image_mask] = image_embeds  # 将图像嵌入向量替换到输入嵌入张量的相应位置
            
            # 处理输入的视频， 移动到和 inputs_embeds 相同的位置
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
                
            # 将注意力掩码张量移动到与输入嵌入张量相同的设备上
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 调用模型的前向传播方法，得到模型的输出
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 最后一层隐藏层张量
        hidden_states = outputs[0] # 获取模型输出的隐藏状态
        
        # 为什么要这个时候重新设置 batch_size，切换成数据集实际的 batch_size
        if has_hard_negative: 
            batch_size = len(hidden_states) // 3 # 样本，正样本，困难负样本
        
        # 如果不是推理模式，将批量大小设置为隐藏状态数量的二分之一
        elif not inference: 
            batch_size = len(hidden_states) // 2 # 样本，正样本，（随机负样本呢）
        
        # 如果是推理模式，将批量大小设置为隐藏状态的数量    
        elif inference:            
            batch_size = len(hidden_states)      # 样本
            
        # 在推理模式下，确保批量大小与隐藏状态数量一致
        if inference:    
            assert batch_size == len(hidden_states)

        # 获取嵌入 token 的 ID
        embed_index = self.config.emb_token_ids[0]
        # 找出每个样本中嵌入 token 的位置： labels：[batch_size, seq_len]
        # argmax 返回 labels 的 第 2 个维度 最大值的位置 embed_indices： [ batch_size]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
        # 根据嵌入 token 的位置提取相应的嵌入特征
        # 进行索引 通过 [torch.arange(len(embed_indices)), embed_indices - 1]
        # 默认嵌入 token 的前一个位置作为 embedding 的特征
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)

        if inference:
            
            if ids is not None: 
                # 在推理模式下，如果提供了通用 ID 列表，返回嵌入特征和 ID 列表
                return embed_features, ids
                
            elif qids is not None or dids is not None: # 针对 M-BEIR 数据集
                # 如果提供了查询 ID 或文档 ID 列表，返回嵌入特征和相应的 ID 列表
                return embed_features, qids, dids
                
            # 否则，只返回嵌入特征
            return embed_features
            
        if has_hard_negative:
            # 如果有难负样本，将嵌入特征分为三组
            # 为了便于阅读添加了括号 chaxjli
            embed1, embed2, embed3 = (embed_features[:batch_size], 
                                      embed_features[batch_size:2*batch_size], embed_features[2*batch_size:])
        else:
            # 否则，将嵌入特征分为两组
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]
        # 初始化交叉熵损失函数
        loss_fct = nn.CrossEntropyLoss()
        # 跨 GPU 配合
        if dist.is_initialized():
            # 如果分布式训练环境已初始化
            if has_hard_negative: # 这个是如何判断的 ？？？？？
                # 如果有难负样本，收集所有进程中的难负样本嵌入特征
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                # dist.all_gather 操作本身不会将当前进程的 embed3 直接放入 embed3_list 中对应自己进程索引的位置
                # 而是放入其他进程的 embed3_list 中对应自己进程索引的位置
                embed3_list[dist.get_rank()] = embed3  # 所以需要重新赋值一下
                embed3 = torch.cat(embed3_list, 0)
            
            # 初始化用于收集嵌入特征的列表
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            # 收集所有进程中的嵌入特征
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())

            # 由于 all_gather 结果没有梯度，将当前进程的原始嵌入特征替换到收集列表中
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            # 将收集到的嵌入特征拼接成完整的批量嵌入特征
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        # 初始化相似度计算模块
        sim = Similarity(temp=0.05)

        # 对嵌入特征进行归一化处理
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        # 计算嵌入特征之间的相似度矩阵
        cos_sim = sim(embed1.unsqueeze(1), embed2.unsqueeze(0))

        if has_hard_negative:
            # 如果有难负样本，计算嵌入特征与难负样本之间的相似度矩阵
            embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
            # 将两个相似度矩阵拼接在一起
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
        
        # 生成标签，用于计算交叉熵损失
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        # 计算交叉熵损失
        loss = loss_fct(cos_sim, nce_labels)
        # 返回分类任务的输出，包含损失值
        return SequenceClassifierOutput(loss=loss)

    def inference(
        self,
        # 输入的 token ID 张量
        input_ids: torch.LongTensor = None,
        # 注意力掩码张量，可选
        attention_mask: Optional[torch.Tensor] = None,
        # 位置 ID 张量，可选
        position_ids: Optional[torch.LongTensor] = None,
        # 过去的键值对列表，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签张量，可选
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存，可选
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可选
        return_dict: Optional[bool] = None,
        # 图像像素值张量，可选
        pixel_values: Optional[torch.Tensor] = None,
        # 视频像素值张量，可选
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        # 图像网格的时间、高度、宽度张量，可选
        image_grid_thw: Optional[torch.LongTensor] = None,
        # 视频网格的时间、高度、宽度张量，可选
        video_grid_thw: Optional[torch.LongTensor] = None,
        # RoPE 偏移量张量，可选
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        # 如果未指定 output_attentions，则使用模型配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 如果输入嵌入张量未提供，则通过词嵌入层将输入的 token ID 转换为嵌入向量
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                # 将图像像素值的类型转换为视觉模块所需的类型
                pixel_values = pixel_values.type(self.visual.get_dtype())
                # 通过视觉模块将图像像素值转换为嵌入向量
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                # 找出输入中图像 token 的位置
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    # 在训练模式下，克隆输入嵌入张量，避免原地修改
                    inputs_embeds = inputs_embeds.clone()
                # 将图像嵌入向量替换到输入嵌入张量的相应位置
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                # 将视频像素值的类型转换为视觉模块所需的类型
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                # 通过视觉模块将视频像素值转换为嵌入向量
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                # 找出输入中视频 token 的位置
                video_mask = input_ids == self.config.video_token_id
                # 将视频嵌入向量替换到输入嵌入张量的相应位置
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                # 将注意力掩码张量移动到与输入嵌入张量相同的设备上
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 调用模型的前向传播方法，得到模型的输出
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]
        # 获取批量大小，这里直接使用隐藏状态的数量
        batch_size = len(hidden_states)
        # 从模型配置中获取嵌入标记的 ID
        embed_index = self.config.emb_token_ids[0]
        # 在输入的 token ID 中找出嵌入标记所在的位置索引
        embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1) 
        # 根据索引从隐藏状态中提取嵌入特征
        # 这里取嵌入标记前一个位置的隐藏状态作为嵌入特征
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]  # (batch_size, embed_dim)
        # 对嵌入特征进行 L2 归一化，确保特征向量的模长为 1
        embed_features = F.normalize(embed_features, dim=-1)
        # 返回归一化后的嵌入特征
        return embed_features


