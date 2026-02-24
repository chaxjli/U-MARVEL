from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging
from models.loss_function import isotropy_loss
from models.FocalInfoNCELoss import FocalInfoNCELoss,FocalInfoNCEABSLoss
from models.LLaVELoss import LLaVELoss
from models.DiHTLoss import DiHTLoss
from models.SoftCSELoss import SoftCSELoss_Weight
from models.SoftCSELoss import SoftCSELoss_Temperature
logger = logging.get_logger(__name__)
from models.latent_attention_block import LatentAttentionBlock
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from models.qwen2_vl_bidirectional_atten import BiQwen2VLForConditionalGeneration
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp=0.05):
        super().__init__()
        # self.temp = temp
        # self.cos = nn.CosineSimilarity(dim=-1)
        self.temp = nn.Parameter(torch.tensor(temp))  # self.temperature = 0.05
    def forward(self, x, y):
        return x @ y.T / self.temp

# 在保存模型时，只有被初始化并且不是 None 的属性才会被保存
# 定义一个继承自 BiQwen2VLForConditionalGeneration 的类，用于 Qwen2-VL 模型的微调
class Qwen2VLRetFinetuneHardNegForConditionalGeneration(BiQwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mean_pooling = True                 # 是否使用全局平局池化
        self.use_bi_atten = True                 # 是否使用双向注意
        self.use_instruction_mask = False        # 是否使用指令 mask
        self.use_bi_loss= False                  # 是否使用双向损失, 默认不使用
        self.use_self_attent_pooling = False     # 是否使用自注意力池化，默认不使用，自注意力池化和全局平均池化互斥
        self.use_latent_atten = False            # 是否使用潜在注意力模块
        self.use_isotropy_loss = False           # 是否使用同构损失, 默认不使用
        self.use_feature_constraint = False      # 是否使用特征约束
        self.use_rerank_scores = False               # 是否使用 rerank 模型的 scores

        # 自注意力池化和全局平均池化互斥
        assert sum([self.use_self_attent_pooling,self.mean_pooling]) <= 1, \
            "自注意力池化和全局平均池化互斥, 不能同时选择。"
        
        self.use_cross_entropy_loss = True          # 是否使用交叉熵损失
        self.use_focal_infonce_loss = False         # 是否使用焦点损失
        self.use_focal_infonce_abs_loss = False     # 是否使用绝对值焦点损失
        self.use_diht_loss = False                  # 是否使用 DIHT 损失
        self.use_llave_loss = False                 # 是否使用 LLaVE 损失
        self.use_softcse_weight_loss = False        # 是否使用 SoftCSE 损失，个性化权重
        self.use_softcse_temperature_loss = False   # 是否使用 SoftCSE 损失，个性化温度
        
        # 确保只有一个损失函数被启用, 如果启用多个损失函数，则抛出异常，默认使用交叉熵损失
        # 如果使用非交叉熵损失，则需要在训练时指定 use_cross_entropy_loss=False
        assert sum([self.use_cross_entropy_loss, self.use_focal_infonce_loss, 
                    self.use_diht_loss,self.use_llave_loss,self.use_focal_infonce_abs_loss,
                    self.use_softcse_weight_loss, self.use_softcse_temperature_loss]) == 1, \
            "Only one loss function can be set to True."
        self.loss_fct = nn.CrossEntropyLoss() # 默认使用交叉熵损失

        # 定义双向损失的权重
        self.querytocand = 0.5
        self.candtoquery = 0.5
        rank0_print("双向损失的权重: ", self.querytocand, self.candtoquery)
        
        # 定义 hard 样本的数量
        self.topk_hard_negative = 5
        self.topk_modality_hard_negative = 5
        self.ignore_batch_other_samples = False  # Ignore other negative samples within the batch

        self.sim = Similarity(temp=0.05) # 定义一个相似度计算的类
        

    # 初始化损失函数，验证有且仅有一个损失函数被启用
    def _initialize_loss_functions(self):
        # 定义参与校验的损失函数列表
        LOSS_FUNCTIONS = [
            self.use_cross_entropy_loss,
            self.use_focal_infonce_loss,
            self.use_focal_infonce_abs_loss,
            self.use_diht_loss,
            self.use_llave_loss,
            self.use_softcse_weight_loss,
            self.use_softcse_temperature_loss,
        ]
        # 确保只有一个损失函数被启用
        assert sum(LOSS_FUNCTIONS) == 1, "Only one loss function can be set to True."
        if self.use_cross_entropy_loss:
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.use_focal_infonce_loss:
            self.loss_fct = FocalInfoNCELoss()
        elif self.use_focal_infonce_abs_loss:
            self.loss_fct = FocalInfoNCEABSLoss()
        elif self.use_diht_loss:
            self.loss_fct = DiHTLoss()
        elif self.use_llave_loss:
            self.loss_fct = LLaVELoss()
        elif self.use_softcse_weight_loss:
            self.loss_fct = SoftCSELoss_Weight()
        elif self.use_softcse_temperature_loss:
            self.loss_fct = SoftCSELoss_Temperature()
        else:
            raise ValueError("No loss function is set.")
    
    def _initialize_latent_attention(self):
        """
        初始化潜在注意力模块
        """
        assert hasattr(self.config, "hidden_size"), "hidden_size 属性不存在于配置中"
        self.latent_dim_scale = 1  # 如果使用潜在注意力模块，请指定 latent_dim_scale = latent_dim/hidden_dim，默认使用 1
        hidden_dim = self.config.hidden_size
        latent_dim = int(hidden_dim * self.latent_dim_scale)
        
        self.latent_attention = LatentAttentionBlock(latent_dim=latent_dim, hidden_dim=hidden_dim)
        rank0_print("LatentAttentionBlock 初始化完成")
        rank0_print(f"潜在注意力模块尺寸: {self.latent_attention.latent_array.size()}")

    
    # 确认一下这个函数到底在哪里使用了 ？？？
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
        instruction_mask: Optional[torch.Tensor] = None, # 指令 mask
        feature_list: Optional[List[torch.Tensor]] = None, # 特征约束
        scores_list: Optional[List[float]] = None  # 如果存储的是浮点数分数
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

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

        hidden_states = outputs[0]
        if self.use_latent_atten: # 如果使用潜在注意力模块
            hidden_states = self.latent_attention(hidden_states)
        
        # 如果使用指令 mask ---------------------------------------------------------
        if self.use_instruction_mask:
            if labels.shape != instruction_mask.shape:
                raise ValueError("labels 和 instruction_mask 的维度不匹配。")
            else:
                # 将 instruction_mask 不为 0 的地方对应的 labels 位置设置成 -100
                labels[instruction_mask != 0] = -100
        # -------------------------------------------------------------------------

        # 平均池化
        if self.mean_pooling:
            embed_features = self._global_mean_pool(hidden_states, labels)
        else:
            embed_index = self.config.emb_token_id # 用于提取特征的 token, 原来是 self.config.emb_token_ids[0]
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1)
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        
        return embed_features 
    
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
        instruction_mask: Optional[torch.Tensor] = None, # 指令 mask
        inference=False,
        has_hard_negative=False, # 是否使用 hard negative
        has_modality_hard_negative=False, # 是否使用 modality hard negative
        feature_list: Optional[List[torch.Tensor]] = None, # 特征约束
        scores_list: Optional[List[float]] = None,  # 如果存储的是浮点数分数
        qids=None,
        dids=None,
        ids=None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set mini_batch to 32
        mini_batch_size = 32 
        input_ids_list = torch.split(input_ids, mini_batch_size)
        attention_mask_list = torch.split(attention_mask, mini_batch_size)
        if image_grid_thw is not None:
            cumsum_pixel_values = torch.cumsum(image_grid_thw[:, 1] * image_grid_thw[:, 2], dim=-1) 
            zero_tensor = torch.tensor([0], device=cumsum_pixel_values.device) # be convinient for extracting batch_pixel_values
            cumsum_pixel_values = torch.cat((zero_tensor, cumsum_pixel_values))
            image_nums = 0
        
        all_hidden_states = []

        for i in range(len(input_ids_list)):
            if inputs_embeds is None:
                batch_inputs_embeds = self.model.embed_tokens(input_ids_list[i])
                if pixel_values is not None:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                    if current_image_num != 0:
                        batch_pixel_values = pixel_values[cumsum_pixel_values[image_nums] : cumsum_pixel_values[image_nums + current_image_num]]
                        batch_pixel_values = batch_pixel_values.type(self.visual.get_dtype())
                        batch_image_embeds = self.visual(batch_pixel_values, grid_thw=image_grid_thw[image_nums:image_nums + current_image_num]).to(batch_inputs_embeds.device)
                        image_nums = image_nums + current_image_num
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                    video_mask = input_ids == self.config.video_token_id
                    inputs_embeds[video_mask] = video_embeds
                if attention_mask is not None:
                    batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)        
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

            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)
                
        # 将所有的 hidden_states 拼接在一起-----------------------------------------------
        hidden_states = torch.cat(all_hidden_states)
        # 根据 has_hard_negative 和 has_modality_hard_negative 确定 query 的 batch_size
        # 这个 batch_size 就是 query 的 batch_size
        col_num = 2
        if has_hard_negative:
            col_num += self.topk_hard_negative
        if has_modality_hard_negative:
            col_num += self.topk_modality_hard_negative
        if inference:
            batch_size = len(hidden_states)
        elif not inference:
            assert len(hidden_states) % col_num == 0, "hidden_states 的长度必须是 col_num 的整数倍, hidden_states 的长度: {}, col_num: {}".format(len(hidden_states), col_num)
            batch_size = len(hidden_states) // col_num
        if inference:
            assert batch_size == len(hidden_states)
        # ---------------------------------------------------------------------------
        # 如果使用潜在注意力模块
        if self.use_latent_atten:
            assert self.latent_attention is not None, "LatentAttentionBlock is not initialized"
            hidden_states = self.latent_attention(hidden_states)
        # 如果使用指令 mask ---------------------------------------------------------
        if self.use_instruction_mask:
            if labels.shape != instruction_mask.shape:
                raise ValueError("labels 和 instruction_mask 的维度不匹配。")
            else:
                # 将 instruction_mask 不为 0 的地方对应的 labels 位置设置成 -100
                labels[instruction_mask != 0] = -100
        # -------------------------------------------------------------------------
        # 平均池化
        if self.mean_pooling:
            embed_features = self._global_mean_pool(hidden_states, labels)
        else:
            embed_index = self.config.emb_token_id # 用于提取特征的 token, 原来是 self.config.emb_token_ids[0]
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        if inference: # 代码运行到这里，推理阶段要进行返回值
            if ids is not None:
                return embed_features, ids 
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features 
        # # --------------------------------------------------------------------------------------------------------------------------------------
        # 下面的代码只有在训练阶段才会执行, 忽不忽略都要进行下面的代码
        # 如果使用特征约束 -----------------------------------------------------------------------------------------------------------
        if self.use_feature_constraint:
            assert feature_list is not None, "使用特征约束时，feature_list 不能为空"
            feartures = torch.stack(feature_list, dim=0)
            assert feartures.requires_grad == False, "特征约束的 requires_grad 属性应该为 False, 这个不计算梯度"
            # 0.1 是一个超参数，表示特征约束的权重，compute_cos_sim_ignore_batch_other_samples 计算的余弦相似度
            # 这里的特征约束损失是一个正则化项，表示当前的特征和之前的特征之间的差异，这个损失函数的作用是让模型的特征更加稳定，避免过拟合
            feature_constraint_loss = 0.1*compute_cos_sim_ignore_batch_other_samples(F.normalize(hidden_states.mean(dim=1), dim=-1), F.normalize(feartures, dim=-1)).mean()
            if dist.is_initialized():
                dist.all_reduce(feature_constraint_loss, op=dist.ReduceOp.SUM)
                feature_constraint_loss /= dist.get_world_size()
                feature_constraint_loss.to(torch.bfloat16)
                rank0_print(f"特征约束的损失: {feature_constraint_loss}")
        
        # 对 query 和 pos 进行特征提取，忽不忽略都要进行这一步
        embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:2*batch_size]
        if dist.is_initialized():
            # 如果使用 hard negative , 处理 embed3 ------------------------------------------------------------------
            if has_hard_negative:
                embed_topk_hard_negative = []  # 用来保存 top k 个 hard negative 的 embed（embed 的数量是 batch_size 个）
                for i in range(self.topk_hard_negative):
                    embed3 = embed_features[(2+i)*batch_size:(2+i+1)*batch_size]
                    # 如果不忽略 batch 其他负样本，需要进行 all_gather 操作
                    if not self.ignore_batch_other_samples:
                        embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                        dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                        embed3_list[dist.get_rank()] = embed3 
                        embed3 = torch.cat(embed3_list, 0)
                    embed_topk_hard_negative.append(embed3)

            # 如果使用 modality hard negative , 处理 embed4 -------------------------------------------------------
            if has_modality_hard_negative:
                embed_topk_modality_hard_negative = []
                for i in range(self.topk_modality_hard_negative):
                    start_index = 2 + self.topk_hard_negative + i if has_hard_negative else 2 + i
                    embed4 = embed_features[start_index*batch_size:(start_index+1)*batch_size]
                    # 如果不忽略 batch 其他负样本，需要进行 all_gather 操作
                    if not self.ignore_batch_other_samples:
                        embed4_list = [torch.zeros_like(embed4) for _ in range(dist.get_world_size())]
                        dist.all_gather(tensor_list=embed4_list, tensor=embed4.contiguous())
                        embed4_list[dist.get_rank()] = embed4 
                        embed4 = torch.cat(embed4_list, 0)
                    embed_topk_modality_hard_negative.append(embed4)
            
            # 如果不忽略 batch 其他负样本，需要进行对 embed1 和 embed2 进行 all_gather 操作
            if not self.ignore_batch_other_samples: 
                embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
                embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
                dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())
                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                embed1_list[dist.get_rank()] = embed1
                embed2_list[dist.get_rank()] = embed2
                # Get full batch embeddings: (bs x N, hidden)
                embed1 = torch.cat(embed1_list, 0)
                embed2 = torch.cat(embed2_list, 0)
        
        embed1 = F.normalize(embed1, dim=-1) # 对 embed1 进行归一化
        embed2 = F.normalize(embed2, dim=-1) # 对 embed2 进行归一化
        if hasattr(self.sim, "modules_to_save"):
            temp = self.sim.modules_to_save["default"].temp
        else:
            temp = self.sim.temp
        # rank0_print(f"当前的温度参数: {temp}")
        # 代码运行到这里，忽不忽略 batch 其他负样本都要进行上面的代码，下面要开始去分成两大块 --------------------------------------------------------
        # 如果忽略 batch 其他负样本，只考虑正样本和困难负样本的相似度 ***************************************************************
        if self.ignore_batch_other_samples:
            cos_sim = compute_cos_sim_ignore_batch_other_samples(embed1, embed2) # 计算 query 和正样本 pos 的相似度
            if self.use_bi_loss: # 如果使用双向损失
                inverse_cos_sim = compute_cos_sim_ignore_batch_other_samples(embed2, embed1)
            if has_hard_negative:
                for embed3 in embed_topk_hard_negative:
                    embed3 = F.normalize(embed3, dim=-1) # 对 embed3 进行归一化
                    embed1_embed3_cos = compute_cos_sim_ignore_batch_other_samples(embed1, embed3)
                    cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
                    if self.use_bi_loss:
                        embed2_embed3_cos = compute_cos_sim_ignore_batch_other_samples(embed2, embed3)
                        inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed3_cos], 1)
            if has_modality_hard_negative:
                for embed4 in embed_topk_modality_hard_negative:
                    embed4 = F.normalize(embed4, dim=-1)
                    embed1_embed4_cos = compute_cos_sim_ignore_batch_other_samples(embed1, embed4)
                    cos_sim = torch.cat([cos_sim, embed1_embed4_cos], 1)
                    if self.use_bi_loss:
                        embed2_embed4_cos = compute_cos_sim_ignore_batch_other_samples(embed2, embed4)
                        inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed4_cos], 1)
            assert cos_sim.size(0) == batch_size, f"cos_sim.size(0) != batch_size, cos_sim.size(0): {cos_sim.size(0)}, batch_size: {batch_size}"
            assert cos_sim.size(1) == (col_num - 1), f"cos_sim.size(1) != (col_num - 1), cos_sim.size(1): {cos_sim.size(1)}, col_num: {col_num}"
            
            # 生成一个长度为 cos_sim.size(0) 且全为 0 的张量，因为 pos 的位置是 0
            nce_labels = torch.zeros(cos_sim.size(0), dtype=torch.long).to(cos_sim.device)
            if (cos_sim < -1).any() or (cos_sim > 1).any():
                rank0_print(f"发生错误, cos_sim 不在 [-1, 1] 之间")
            # 计算正向损失 ---------------------------------------------------------------------------
            if self.use_cross_entropy_loss: # 如果使用交叉熵损失,手动除以温度参数
                cos_sim = cos_sim / temp
            loss = self.loss_fct(cos_sim, nce_labels)  # (1,) 计算损失
            if self.use_bi_loss:
                if (inverse_cos_sim < -1).any() or (inverse_cos_sim > 1).any():
                    rank0_print(f"发生错误, inverse_cos_sim 不在 [-1, 1] 之间")
                if self.use_cross_entropy_loss:
                    inverse_cos_sim = inverse_cos_sim / temp
                inverse_loss = self.loss_fct(inverse_cos_sim, nce_labels)
                loss = self.querytocand * loss + self.candtoquery * inverse_loss
            
            if dist.is_initialized(): # 聚合损失求平均
                loss.float() # 转换为 float 类型
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                loss.to(torch.bfloat16)
            
            if self.use_feature_constraint: # 如果使用特征约束
                loss += feature_constraint_loss
            
            return SequenceClassifierOutput(loss=loss)
        # 如果没有忽略 batch 其他负样本，计算余弦相似度   *************************************************************
        else:
            cos_sim = embed1 @ embed2.T         # (B, B) 矩阵乘法
            if self.use_bi_loss:                # 如果使用双向损失
                inverse_cos_sim = cos_sim.T     # 直接转置即可
            # 如果使用 hard negative
            if has_hard_negative:
                for embed3 in embed_topk_hard_negative:
                    embed3 = F.normalize(embed3, dim=-1)
                    embed1_embed3_cos = (embed1 @ embed3.T)
                    cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
                    if self.use_bi_loss: # 如果使用双向损失
                        embed2_embed3_cos = (embed2 @ embed3.T)
                        inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed3_cos], 1)
            # 如果使用 modality hard negative
            if has_modality_hard_negative:
                for embed4 in embed_topk_modality_hard_negative:
                    embed4 = F.normalize(embed4, dim=-1)
                    embed1_embed4_cos = (embed1 @ embed4.T)
                    cos_sim = torch.cat([cos_sim, embed1_embed4_cos], 1)
                    if self.use_bi_loss: # 如果使用双向损失
                        embed2_embed4_cos = (embed2 @ embed4.T)
                        inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed4_cos], 1)
                
            nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            assert cos_sim.size(0) == batch_size*(dist.get_world_size() if dist.is_initialized() else 1),"发生维度错误, cos_sim.size(0): {}, batch_size: {}".format(cos_sim.size(0), batch_size)
            assert cos_sim.size(1) == (col_num-1)*cos_sim.size(0), "发生维度错误, cos_sim.size(1): {}, col_num: {}".format(cos_sim.size(1), col_num)
            # 检查一下 cos_sim 是否位于 [-1,1] 之间
            if (cos_sim < -1).any() or (cos_sim > 1).any():
                rank0_print(f"发生错误, cos_sim 不在 [-1, 1] 之间")
            # rank0_print(f"cos_sim 最原始的相似的矩阵: {cos_sim}")
            
            #  计算正向损失 ---------------------------------------------------------------------------------------------------------------------
            if self.use_cross_entropy_loss: # 如果使用交叉熵损失,手动除以温度参数
                cos_sim = cos_sim / temp
            loss = self.loss_fct(cos_sim, nce_labels)  # (1,) 计算损失
            # 计算反向损失 ---------------------------------------------------------------------------------------------------------------------
            if self.use_bi_loss: # 如果使用双向损失
                if (inverse_cos_sim < -1).any() or (inverse_cos_sim > 1).any():
                    rank0_print(f"发生错误, inverse_cos_sim 不在 [-1, 1] 之间")
                if self.use_cross_entropy_loss: # 如果使用交叉熵损失
                    inverse_cos_sim = inverse_cos_sim / temp
                inverse_loss = self.loss_fct(inverse_cos_sim, nce_labels)
                loss = self.querytocand * loss + self.candtoquery * inverse_loss
            
            if self.use_feature_constraint: # 如果使用特征约束
                loss += feature_constraint_loss

            return SequenceClassifierOutput(loss=loss)

def compute_cos_sim_ignore_batch_other_samples(embed1, embed2):
    """
    计算余弦相似度（优化版）
    :param embed1: (B, D) 已归一化
    :param embed2: (B, D) 已归一化
    :return: (B, 1) 对应位置的余弦相似度
    """
    # 计算前检查 embed1 和 embed2 形状是否一致
    if embed1.shape != embed2.shape:
        raise ValueError(f"计算相似度两个 embed 的形状不一致: {embed1.shape} vs {embed2.shape}")
    # 计算余弦相似度
    cos_sim = (embed1 * embed2).sum(dim=1, keepdim=True)  # (B, 1)
    return cos_sim

