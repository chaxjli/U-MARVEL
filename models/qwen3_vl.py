from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling

from models.loss_function import isotropy_loss
from models.FocalInfoNCELoss import FocalInfoNCELoss,FocalInfoNCEABSLoss
from models.LLaVELoss import LLaVELoss
from models.DiHTLoss import DiHTLoss
from models.SoftCSELoss import SoftCSELoss_Weight
from models.SoftCSELoss import SoftCSELoss_Temperature
logger = logging.get_logger(__name__)
from models.latent_attention_block import LatentAttentionBlock
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, PreTrainedTokenizer
from models.qwen3_vl_bidirectional_atten import BiQwen3VLForConditionalGeneration,Qwen3VLCausalLMOutputWithPast
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.generic import check_model_inputs
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
        self.temp = nn.Parameter(torch.tensor(temp))
    def forward(self, x, y):
        return x @ y.T / self.temp
# 在保存模型时，只有被初始化并且不是 None 的属性才会被保存
# 定义一个继承自 BiQwen3VLForConditionalGeneration 的类，用于 Qwen3-VL 模型的微调
class Qwen3VLRetForConditionalGeneration(BiQwen3VLForConditionalGeneration):
    def __init__(self, config,use_bi_atten=True,temp=0.05):
        super().__init__(config,use_bi_atten=use_bi_atten)
        self.mean_pooling = True                 # 是否使用全局平局池化
        self.use_bi_atten = use_bi_atten         # 是否使用双向注意
        self.use_latent_atten = False            # 是否使用潜在注意力模块
        self.use_instruction_mask = False        # 是否使用指令 mask
        self.use_bi_loss= False                  # 是否使用双向损失, 默认不使用
        self.use_isotropy_loss = False           # 是否使用同构损失, 默认不使用
        self.use_self_attent_pooling = False     # 是否使用自注意力池化，默认不使用，自注意力池化和全局平均池化互斥
        self.use_feature_constraint = False      # 是否使用特征约束
        self.rerank_scores = False               # 是否使用 rerank 模型的 scores
        
        # 自注意力池化和全局平均池化互斥
        assert sum([self.use_self_attent_pooling,self.mean_pooling]) <= 1, \
            "自注意力池化和全局平均池化互斥, 不能同时选择。"
        
        if self.use_isotropy_loss:
            self.lambda_iso = 5e2                # 同构损失的权重, lambda_iso: float = 5e1 默认
            rank0_print("同构损失的权重: ", self.lambda_iso)
        
        self.use_cross_entropy_loss = True          # 是否使用交叉熵损失
        self.use_focal_infonce_loss = False         # 是否使用焦点损失
        self.use_focal_infonce_abs_loss = False     # 是否使用绝对值焦点损失
        self.use_diht_loss = False                  # 是否使用 DIHT 损失
        self.use_llave_loss = False                 # 是否使用 LLaVE 损失
        self.use_softcse_weight_loss = False        # 是否使用 SoftCSE 损失，个性化权重
        self.use_softcse_temperature_loss = False   # 是否使用 SoftCSE 损失，个性化温度
        self.temp = temp                            # 温度参数初始化
        
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

        self.sim = Similarity(temp=self.temp)  # 使用自定义的相似度计算模块
        rank0_print("模型初始化打印温度参数:", self.sim.temp)


    
    # 验证有且仅有一个损失函数被启用
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

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        instruction_mask: Optional[torch.Tensor] = None, # 指令 mask
        inference=False,
        has_hard_negative=False, # 是否使用 hard negative
        has_modality_hard_negative=False, # 是否使用 modality hard negative
        feature_list: Optional[List[torch.Tensor]] = None, # 特征约束
        scores_list: Optional[List[float]] = None,  # 如果存储的是浮点数分数
        qids=None,
        dids=None,
        ids=None,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds,deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, deepstack_video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_embeds = video_embeds.to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
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
        if self.use_instruction_mask and instruction_mask is not None: # 这里其实不需要 instruction_mask is not None 但是实验 merged model 推理会有问题
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
        
        # # -------------------------------------------------------
        if inference:
            # rank0_print("qwen3_vl_finetune 的 ids 的长度：", len(ids))
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
        
        # add normalization
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)
        cos_sim = self.sim(embed1, embed2)  # (B, B) 矩阵乘法
        
        if self.use_bi_loss: # 如果使用双向损失
            inverse_cos_sim = cos_sim.T  # 直接转置即可
        
        
        # 如果使用 hard negative 或者 modality hard negative，处理 embed3
        if has_hard_negative or has_modality_hard_negative:
            embed3 = F.normalize(embed3, dim=-1)
            # embed1_embed3_cos = (embed1 @ embed3.T)
            embed1_embed3_cos = self.sim(embed1, embed3)
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)

            if self.use_bi_loss: # 如果使用双向损失
                # embed2_embed3_cos = (embed2 @ embed3.T)
                embed2_embed3_cos = self.sim(embed2, embed3)
                inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed3_cos], 1)
            

        # 如果使用 hard negative 和 modality hard negative，处理 embed4
        if has_hard_negative and has_modality_hard_negative:
            embed4 = F.normalize(embed4, dim=-1)
            embed1_embed4_cos = self.sim(embed1, embed4)
            cos_sim = torch.cat([cos_sim, embed1_embed4_cos], 1)
            
            if self.use_bi_loss: # 如果使用双向损失
                embed2_embed4_cos = self.sim(embed2, embed4)
                inverse_cos_sim = torch.cat([inverse_cos_sim, embed2_embed4_cos], 1)
            
        
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        
        if hasattr(self.sim, "modules_to_save"):
            temp = self.sim.modules_to_save["default"].temp
            # rank0_print("从 modules_to_save 中获取的温度参数: ", temp.item(),temp.requires_grad)
        else:
            temp = self.sim.temp
            # rank0_print("从 sim 中获取的温度参数: ", temp.item(),temp.requires_grad)
        
        #  计算正向损失 ---------------------------------------------------------------------------------------------------------------------
        loss = self.loss_fct(cos_sim, nce_labels)  # (1,) 计算损失
        
        # 计算反向损失 ---------------------------------------------------------------------------------------------------------------------
        if self.use_bi_loss: # 如果使用双向损失
            inverse_loss = self.loss_fct(inverse_cos_sim, nce_labels)
            # rank0_print(f"正向损失: {loss}")
            # rank0_print(f"反向损失: {inverse_loss}")
            loss = self.querytocand * loss + self.candtoquery * inverse_loss
            # rank0_print(f"双向损失: {loss}")
                       
        return SequenceClassifierOutput(loss=loss)