from typing import Tuple, Optional, List, Union 
import torch 
# import torch_npu  # 导入华为昇腾 NPU 相关的 PyTorch 扩展
# from torch_npu.contrib import transfer_to_npu # 适配 npu
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F
from models.qwen2_vl_bidirectional_atten_new import BiQwen2VLForConditionalGeneration
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
# class Qwen2VLRetForConditionalGeneration(BiQwen2VLForConditionalGeneration):
class Qwen2VLRetForConditionalGeneration(BiQwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mean_pooling = True             # 是否使用全局平局池化
        self.use_bi_atten = True             # 是否使用双向注意
        self.use_instruction_mask = True     # 是否使用指令 mask
        self.use_latent_atten = False        # 是否使用潜在注意力模块
        self.use_bi_loss= False              # 是否使用双向损失, 默认不使用
        if self.use_latent_atten:            # 如果使用潜在注意力模块，
            self.latent_dim_scale = 1        # 请指定 latent_dim_scale = latent_dim/hidden_dim, 默认使用 1
            self.latent_attention = None     # 使用的时候根据 decoder_output 的维度初始化
        if hasattr(self.model, "mean_pooling"):
            rank0_print("self.model 存在 mean_pooling 属性")
        else:
            rank0_print("self.model 不存在 mean_pooling 属性")
        
        if hasattr(self.model, "use_bi_atten"):
            rank0_print("self.model 存在 use_bi_atten 属性")
            self.model.use_bi_atten = self.use_bi_atten
        else:
            rank0_print("self.model 不存在 use_bi_atten 属性")
    
    # 修改 model 的 use_bi_atten 属性
    def _set_model_use_bi_atten(self):
        
        rank0_print("mean_pooling: ", self.mean_pooling)
        rank0_print("use_bi_atten: ", self.use_bi_atten)
        rank0_print("use_instruction_mask: ", self.use_instruction_mask)

        if hasattr(self.model, "mean_pooling"):
            rank0_print("self.model 存在 mean_pooling 属性,其值为：", self.model.mean_pooling)
        else:
            rank0_print("self.model 不存在 mean_pooling 属性")
        if hasattr(self.model, "use_bi_atten"):
            rank0_print("self.model 存在 use_bi_atten 属性，其值为：", self.model.use_bi_atten)
            self.model.use_bi_atten = self.use_bi_atten
        else:
            rank0_print("self.model 不存在 use_bi_atten 属性")
        if hasattr(self.model, "use_instruction_mask"):
            rank0_print("self.model 存在 use_instruction_mask 属性，其值为：", self.use_instruction_mask)
        else:
            rank0_print("self.model 不存在 use_instruction_mask 属性")
        

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
        has_hard_negative=False, # 是否使用 hard negative
        has_modality_hard_negative=False, # 是否使用 modality hard negative
        qids=None,
        dids=None,
        ids=None,
        instruction_mask: Optional[torch.Tensor] = None, # 指令 mask
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
 
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
        # 根据 has_hard_negative 和 has_modality_hard_negative 确定 query 的 batch_size
        if has_hard_negative and has_modality_hard_negative and not inference:
            batch_size = len(hidden_states) // 4
        elif (has_hard_negative or has_modality_hard_negative) and not inference:
            batch_size = len(hidden_states) // 3
        elif not inference:
            batch_size = len(hidden_states) // 2
        elif inference:
            batch_size = len(hidden_states)   
        if inference:
            assert batch_size == len(hidden_states)
        
        # 如果使用指令 mask ---------------------------------------------------------
        if self.use_instruction_mask:
            if instruction_mask is None:
                try:
                    instruction_start_token = self.config.instruction_start_token_id
                    instruction_end_token = self.config.instruction_end_token_id
                    instruction_mask = self.get_instruction_mask(input_ids, instruction_start_token, instruction_end_token)
                    assert labels.shape == instruction_mask.shape, "labels 和 instruction_mask 的维度不匹配。"
                    # rank0_print("labels.shape: ", labels.shape)
                    # rank0_print("instruction_mask.shape: ", instruction_mask.shape)
                    labels[instruction_mask != 0] = -100    # 将 instruction_mask 不为 0 的地方对应的 labels 位置设置成 -100
                except Exception as e:
                    # rank0_print("当前数据不存在 instruction，使用默认的 labels。")
                    # rank0_print("错误信息：", str(e))
                    pass
        # -------------------------------------------------------------------------
        # 平均池化
        if self.mean_pooling:
            embed_features = self._global_mean_pool(hidden_states, labels)
        else:
            embed_index = self.config.emb_token_id
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        if inference:
            embed_features = F.normalize(embed_features, dim=-1)  # 这一步不确定是否需要，看看具体效果再定
            if ids is not None:
                return embed_features, ids 
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features 
        # # -------------------------------------------------------
        if has_hard_negative and has_modality_hard_negative:
            embed1, embed2, embed3, embed4 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:3*batch_size], embed_features[3*batch_size:]
        elif has_hard_negative or has_modality_hard_negative:
            embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
        else:
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]
        
        loss_fct = nn.CrossEntropyLoss()
        if dist.is_initialized():
            # Dummy vectors for allgather
            # 如果使用 hard negative 或者 modality hard negative，处理 embed3
            if has_hard_negative or has_modality_hard_negative:
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                embed3_list[dist.get_rank()] = embed3 
                embed3 = torch.cat(embed3_list, 0)
            # 如果使用 hard negative 和 modality hard negative，处理 embed4
            if has_hard_negative and has_modality_hard_negative:
                embed4_list = [torch.zeros_like(embed4) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed4_list, tensor=embed4.contiguous())
                embed4_list[dist.get_rank()] = embed4
                embed4 = torch.cat(embed4_list, 0)
            # 处理 embed1 和 embed2
            # Dummy vectors for allgather
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            # Get full batch embeddings: (bs x N, hidden)
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        sim = Similarity(temp=0.05)
        loss_fct = nn.CrossEntropyLoss()
        # add normalization
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)
        cos_sim = (embed1 @ embed2.T) / sim.temp  # (B, B) 矩阵乘法
        
        if self.use_bi_loss: # 如果使用双向损失
            inverse_cos_sim = cos_sim.T  # 直接转置即可
        
        # 如果使用 hard negative 或者 modality hard negative，处理 embed3
        if has_hard_negative or has_modality_hard_negative:
            embed3 = F.normalize(embed3, dim=-1)
            embed1_embed3_cos = (embed1 @ embed3.T) / sim.temp
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)

            if self.use_bi_loss: # 如果使用双向损失
                embed2_embed3_cos = (embed2 @ embed3.T) / sim.temp
                inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed3_cos], 1)
            
        # 如果使用 hard negative 和 modality hard negative，处理 embed4
        if has_hard_negative and has_modality_hard_negative:
            embed4 = F.normalize(embed4, dim=-1)
            embed1_embed4_cos = (embed1 @ embed4.T) / sim.temp
            cos_sim = torch.cat([cos_sim, embed1_embed4_cos], 1)
            
            if self.use_bi_loss: # 如果使用双向损失
                embed2_embed4_cos = (embed2 @ embed4.T) / sim.temp
                inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed4_cos], 1)
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss = loss_fct(cos_sim, nce_labels)
        if self.use_bi_loss: # 如果使用双向损失
            loss += loss_fct(inverse_cos_sim, nce_labels)
            loss /= 2
        return SequenceClassifierOutput(loss=loss)

    def inference(
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
        has_hard_negative=False,          # 是否使用 hard negative, 推理阶段不使用
        has_modality_hard_negative=False, # 是否使用 modality hard negative，推理阶段不使用
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
        batch_size = len(hidden_states)
        
        # 如果使用潜在注意力模块而且是推理阶段
        if self.use_latent_atten:
            assert self.latent_attention is not None, "LatentAttentionBlock is not initialized"
            hidden_states = self.latent_attention(hidden_states)

        # 如果使用指令 mask ---------------------------------------------------------
        if self.use_instruction_mask:
            if instruction_mask is None:
                try:
                    instruction_start_token = self.config.instruction_start_token_id
                    instruction_end_token = self.config.instruction_end_token_id
                    instruction_mask = self.get_instruction_mask(input_ids, instruction_start_token, instruction_end_token)
                    assert labels.shape == instruction_mask.shape, "labels 和 instruction_mask 的维度不匹配。"
                    # 将 instruction_mask 不为 0 的地方对应的 labels 位置设置成 -100
                    # rank0_print("labels.shape: ", labels.shape)
                    # rank0_print("instruction_mask.shape: ", instruction_mask.shape)
                    labels[instruction_mask != 0] = -100
                except Exception as e:
                    # rank0_print("获取 instruction_mask 失败，使用默认的 labels。")
                    # rank0_print("错误信息：", str(e))
                    pass
        # -------------------------------------------------------------------------
        
        # 平均池化
        if self.mean_pooling:
            embed_features = self._global_mean_pool(hidden_states, labels)
            embed_features = F.normalize(embed_features, dim=-1)
        else:
            embed_index = self.config.emb_token_id
            embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1) 
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
            embed_features = F.normalize(embed_features, dim=-1)  # 这一步不确定是否需要，看看具体效果再定
        return embed_features 


    # 元宝优化的版本
    def get_instruction_mask(self, input_ids, start_id, end_id):
            # 处理输入为空张量的情况
            if input_ids.numel() == 0:
                return torch.zeros_like(input_ids, dtype=torch.int)
            start_mask = (input_ids == start_id).int()
            end_mask = (input_ids == end_id).int()
            # 累积求和
            start_cum = start_mask.cumsum(dim=1)
            end_cum = end_mask.cumsum(dim=1)
            # 确保开始标记在结束标记之前，避免标记顺序混乱的问题
            valid_mask = start_cum >= end_cum
            cum_mask = (start_cum - end_cum) > 0
            # 处理嵌套情况：通过限制每个开始标记只能对应一个结束标记
            valid_start = start_mask * (valid_mask & (start_cum - end_cum == 1))
            valid_end = end_mask * (valid_mask & (start_cum - end_cum == 0))
            # 生成最终的掩码矩阵
            instruction_mask = (cum_mask | valid_start | valid_end).int().to(input_ids.device)
            return instruction_mask
    def _global_mean_pool(self,hidden_states, labels):
        """
        全局均值池化（所有有效token取平均）
        Args:
            hidden_states: 模型最后一层隐藏状态 [batch_size, seq_len, hidden_dim]
            labels: 每个token的标签 [batch_size, seq_len]（-100表示无效）
        Returns:
            pooled_features: 池化后的特征 [batch_size, hidden_dim]
        """
        # 创建有效掩码（排除-100），并转换为与hidden_states相同的数据类型
        # valid_mask: [batch_size, seq_len, 1]
        valid_mask = (labels != -100).unsqueeze(-1)
        valid_mask = valid_mask.to(hidden_states.dtype)  # 确保数据类型一致

        # 计算加权和（自动广播 valid_mask 到 hidden_dim 维度）
        sum_hidden = torch.sum(hidden_states * valid_mask, dim=1)  # [batch_size, hidden_dim]

        # 计算有效token数量（转换为浮点型）
        num_valid = torch.sum(valid_mask, dim=1)      # [batch_size, 1]
        num_valid = torch.clamp(num_valid, min=1e-7)  # 防止除零

        # 均值池化（广播除法）num_valid: [batch_size, 1]-->[batch_size, hidden_dim]
        pooled_features = sum_hidden / num_valid
        return pooled_features