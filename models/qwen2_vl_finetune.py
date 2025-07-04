from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging
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

# 定义一个继承自 BiQwen2VLForConditionalGeneration 的类，用于 Qwen2-VL 模型的微调
class Qwen2VLRetFinetuneForConditionalGeneration(BiQwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mean_pooling = True                 # 是否使用全局平局池化
        self.use_bi_atten = True                 # 是否使用双向注意
        self.use_latent_atten = False            # 是否使用潜在注意力模块
        self.use_instruction_mask = False        # 是否使用指令 mask
        self.loss_fct = nn.CrossEntropyLoss() # 默认使用交叉熵损失
        self.sim = Similarity(temp=0.05)
    
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
        # ---------------------------------------------------------------------------
        # 如果使用潜在注意力模块
        if self.use_latent_atten:
            assert self.latent_attention is not None, "LatentAttentionBlock is not initialized"
            hidden_states = self.latent_attention(hidden_states)

        # 如果使用指令 mask ---------------------------------------------------------
        if self.use_instruction_mask and instruction_mask is not None:
            if labels.shape != instruction_mask.shape:
                raise ValueError("labels 和 instruction_mask 的维度不匹配。")
            else:
                labels[instruction_mask != 0] = -100
        # -------------------------------------------------------------------------
        # 平均池化
        if self.mean_pooling:
            embed_features = self._global_mean_pool(hidden_states, labels)
        else:
            embed_index = self.config.emb_token_id # 用于提取特征的 token, 原来是 self.config.emb_token_ids[0]
            embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
            embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        
        # # -------------------------------------------------------
        if inference:
            if ids is not None:
                return embed_features, ids 
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features