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
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, PreTrainedTokenizer
from models.qwen2_5_vl_bidirectional_atten import BiQwen2_5_VLForConditionalGeneration,Qwen2_5_VLCausalLMOutputWithPast
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
from transformers.utils.generic import check_model_inputs
from models.distill_rerank_loss_function import (
    compute_kl_divergence,compute_js_divergence, compute_f_divergence,tv_f_function,
    compute_renyi_divergence,compute_generalized_kl_divergence,
    compute_mse_loss,compute_ranking_loss,
)
from transformers.utils import (
    TransformersKwargs, auto_docstring, can_return_tuple, 
    is_torchdynamo_compiling, logging
)

class Similarity(nn.Module):
    
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(temp))  # self.temperature = 0.05

    def forward(self, x, y,sim_compute="dot_product"):
        if sim_compute=="dot_product":
            return x @ y.T / self.temp
        elif sim_compute=="hadamard_product":
            # 计算前检查 x 和 y 形状是否一致
            # cos_sim = (embed1 * embed2).sum(dim=1, keepdim=True)  # (B, 1)
            if x.shape != y.shape:
                raise ValueError(f"计算相似度两个 embed 的形状不一致: {x.shape} vs {y.shape}")
            return (x * y).sum(dim=-1,keepdim=True) / self.temp
        else:
            raise ValueError("sim_compute must be 'dot_product' or 'hadamard_product'")

# 在保存模型时，只有被初始化并且不是 None 的属性才会被保存
# 定义一个继承自 BiQwen2_5_VLForConditionalGeneration 的类，用于 Qwen2_5-VL 模型的微调
class Qwen2_5_VLRetFinetuneDistillRerankForConditionalGeneration(BiQwen2_5_VLForConditionalGeneration):
    def __init__(self, config,use_bi_atten=True,temp=0.05):
        super().__init__(config,use_bi_atten=use_bi_atten)
        self.mean_pooling = True                 # 是否使用全局平局池化
        self.use_bi_atten = use_bi_atten         # 是否使用双向注意
        self.use_instruction_mask = True         # 是否使用指令 mask
        self.use_bi_loss= False                  # 是否使用双向损失, 默认不使用
        self.use_self_attent_pooling = False     # 是否使用自注意力池化，默认不使用，自注意力池化和全局平均池化互斥
        self.use_latent_atten = False            # 是否使用潜在注意力模块
        self.use_isotropy_loss = False           # 是否使用同构损失, 默认不使用
        self.use_feature_constraint = False      # 是否使用特征约束
        self.use_rerank_scores = True            # 是否使用 rerank 模型的 scores
        self.use_distill_with_pos = False         # 是否使用正样本进行辅助蒸馏
        self.use_distill_with_infonce = False      # 是否使用 infonce loss 进行辅助蒸馏

        # 蒸馏时使用infonce loss 进行辅助蒸馏时，必须使用 rerank scores 和 distill with pos
        if self.use_distill_with_infonce:
            assert self.use_rerank_scores, "使用 infonce loss 进行辅助蒸馏时，必须使用 rerank scores"
            assert self.use_distill_with_pos, "使用 infonce loss 进行辅助蒸馏时，必须使用正样本进行辅助蒸馏"
        if self.use_distill_with_pos:
            assert self.use_rerank_scores, "使用正样本进行辅助蒸馏时，必须使用 rerank scores" 

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
        self.temp = temp                            # 温度参数 
        
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
        self.topk_hard_negative = 50
        self.topk_modality_hard_negative = 50
        self.ignore_batch_other_samples = True  # Ignore other negative samples within the batch

        # 定义蒸馏的 rerank 损失函数
        self.use_kl_constraint = False
        self.use_generalized_kl_constraint = False
        self.use_js_constraint = False
        self.use_mse_constraint = False
        self.use_ranking_constraint = False

        self.sim = Similarity(temp=self.temp)  # 使用自定义的相似度计算模块
        rank0_print("模型初始化打印温度参数:", self.sim.temp)

    
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

    @can_return_tuple
    @auto_docstring
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
        pos_index_list: Optional[List[int]] = None,  # 正样本的索引列表
        qids=None,
        dids=None,
        ids=None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens...
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
                Pixel values of images...
            image_grid_thw (`torch.LongTensor` of shape `(batch_size, num_images, 3)`, *optional*):
                Grid size [T, H, W] for each image in the batch.
            video_grid_thw (`torch.LongTensor` of shape `(batch_size, num_videos, 3)`, *optional*):
                Grid size [T, H, W] for each video in the batch.
            instruction_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to indicate instruction tokens in the input.
            inference (`bool`, *optional*, defaults to `False`):
                Whether to run in inference mode.
            has_hard_negative (`bool`, *optional*, defaults to `False`):
                Whether the batch contains hard negative samples.
            has_modality_hard_negative (`bool`, *optional*, defaults to `False`):
                Whether the batch contains modality-level hard negatives.
            feature_list (`List`, *optional*):
                List to collect intermediate features during forward pass.
            scores_list (`List`, *optional*):
                List to collect retrieval scores.
            pos_index_list (`List[int]`, *optional*):
                List to collect positive sample indices.
            qids (`List[str]`, *optional*):
                Query IDs for logging or evaluation.
            dids (`List[str]`, *optional*):
                Document IDs for logging or evaluation.
            ids (`List[str]`, *optional*):
                General sample IDs.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

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
                batch_inputs_embeds = self.model.get_input_embeddings()(input_ids_list[i])
                if pixel_values is not None:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                    if current_image_num != 0:
                        batch_pixel_values = pixel_values[cumsum_pixel_values[image_nums] : cumsum_pixel_values[image_nums + current_image_num]]
                        batch_pixel_values = batch_pixel_values.type(self.visual.dtype)
                        batch_image_embeds = self.visual(batch_pixel_values, grid_thw=image_grid_thw[image_nums:image_nums + current_image_num])
                        batch_image_embeds = batch_image_embeds.to(batch_inputs_embeds.device)

                        image_nums = image_nums + current_image_num
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_embeds= self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    video_embeds = video_embeds.to(batch_inputs_embeds.device)
                    
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
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)
                
        # 将所有的 hidden_states 拼接在一起-----------------------------------------------
        hidden_states = torch.cat(all_hidden_states)
        # 根据 has_hard_negative 和 has_modality_hard_negative 确定 query 的 batch_size
        # 这个 batch_size 就是 query 的 batch_size, 这里不再使用 pos 样本
        col_num = 1
        if has_hard_negative:
            col_num += self.topk_hard_negative
        if has_modality_hard_negative:
            col_num += self.topk_modality_hard_negative
        if self.use_distill_with_pos:
            col_num += 1  # 再加上一个正样本,开始支持使用正样本进行辅助蒸馏
        # ------------------------------------------------------------------------------
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
            # 0.1 是一个超参数，表示特征约束的权重，self.sim(x,y,sim_compute="hadamard_product") 计算的余弦相似度
            # 这里的特征约束损失是一个正则化项，表示当前的特征和之前的特征之间的差异，这个损失函数的作用是让模型的特征更加稳定，避免过拟合
            feature_constraint_loss = 0.1*self.sim(F.normalize(hidden_states.mean(dim=1), dim=-1), F.normalize(feartures, dim=-1), sim_compute="hadamard_product").mean()
        
        if dist.is_initialized():
            # 对 query 进行特征提取，忽不忽略都要进行这一步
            embed1 = embed_features[:batch_size]
            # 如果使用 hard negative , 处理 embed3 ------------------------------------------------------------------
            if has_hard_negative:
                embed_topk_hard_negative = []  # 用来保存 top k 个 hard negative 的 embed（embed 的数量是 batch_size 个）
                for i in range(self.topk_hard_negative):
                    embed3 = embed_features[(1+i)*batch_size:(1+i+1)*batch_size]
                    embed_topk_hard_negative.append(embed3)

            # 如果使用 modality hard negative , 处理 embed4 -------------------------------------------------------
            if has_modality_hard_negative:
                embed_topk_modality_hard_negative = []
                for i in range(self.topk_modality_hard_negative):
                    start_index = 1 + self.topk_hard_negative + i if has_hard_negative else 1 + i
                    embed4 = embed_features[start_index*batch_size:(start_index+1)*batch_size]
                    embed_topk_modality_hard_negative.append(embed4)
            if self.use_distill_with_pos:
                # 处理正样本的 embed2
                embed2 = embed_features[-batch_size:]
        
        
        embed1 = F.normalize(embed1, dim=-1) # 对 embed1 进行归一化
        if hasattr(self.sim, "modules_to_save"):
            temp = self.sim.modules_to_save["default"].temp
            rank0_print("从 modules_to_save 中获取的温度参数: ", temp.item(),temp.requires_grad)
        else:
            temp = self.sim.temp
            rank0_print("从 sim 中获取的温度参数: ", temp.item(),temp.requires_grad)
        # rank0_print(f"当前的温度参数: {temp}")
        # 代码运行到这里，忽不忽略 batch 其他负样本都要进行上面的代码，下面要开始去分成两大块 --------------------------------------------------------
        
        
        # 计算正样本的相似度 --------------------------------------------------------------------------------------------------------
        if self.use_distill_with_pos:
            embed2 = F.normalize(embed2, dim=-1) # 对 embed2 进行归一化
            cos_sim = self.sim(embed1, embed2, sim_compute="hadamard_product")
        else:
            cos_sim = None
        
        # 计算 hard negative 的相似度 ----------------------------------------------------------------------------------------------
        if has_hard_negative:
            for embed3 in embed_topk_hard_negative:
                embed3 = F.normalize(embed3, dim=-1) # 对 embed3 进行归一化
                if cos_sim is None:
                    cos_sim = self.sim(embed1, embed3, sim_compute="hadamard_product")
                else:
                    embed1_embed3_cos = self.sim(embed1, embed3, sim_compute="hadamard_product")
                    cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
        
        if has_modality_hard_negative:
            for embed4 in embed_topk_modality_hard_negative:
                embed4 = F.normalize(embed4, dim=-1)
                if cos_sim is None:
                    cos_sim = self.sim(embed1, embed4, sim_compute="hadamard_product")
                else:
                    embed1_embed4_cos = self.sim(embed1, embed4, sim_compute="hadamard_product")
                    cos_sim = torch.cat([cos_sim, embed1_embed4_cos], 1)

        # 计算蒸馏的分数 ----------------------------------------------------------------------------------------------
        if self.use_rerank_scores:
            assert scores_list is not None, "使用 rerank_scores 时，scores_list 不能为空"
            # 计算 rerank_scores 的损失函数,
            rerank_scores = torch.tensor(scores_list, device=cos_sim.device,requires_grad=False,dtype=cos_sim.dtype)
            rerank_scores = rerank_scores/2  # 除以 2
            assert rerank_scores.requires_grad == False, "rerank_scores 的 requires_grad 属性应该为 False, 这个不计算梯度"
            assert rerank_scores.shape == cos_sim.shape, "rerank_scores 和 cos_sim 的维度不匹配"
 
        # 计算损失函数 ----------------------------------------------------------------------------------------------
        # 检查一下 cos_sim 和 rerank_scores 是否位于 [-1,1] 之间
        if (cos_sim < -1.1/temp).any() or (cos_sim > 1.1/temp).any():
            rank0_print(f"发生错误, cos_sim 不在 [-1, 1] 之间")
            rank0_print("cos_sim 的最小值和最大值是: ", cos_sim.min(), cos_sim.max())
        if (rerank_scores < -1.1/temp).any() or (rerank_scores > 1.1/temp).any():
            rank0_print(f"发生错误, rerank_scores 不在 [-1, 1] 之间")
            rank0_print("rerank_scores 的最小值和最大值是: ", rerank_scores.min(), rerank_scores.max())
        # 对 cos_sim 和 rerank_scores 除以温度参数
        # cos_sim = cos_sim / temp
        rerank_scores = rerank_scores / (temp.detach())
        assert rerank_scores.requires_grad == False, "rerank_scores 的 requires_grad 属性应该为 False, 这个不计算梯度"
        assert cos_sim.requires_grad == True, "cos_sim 的 requires_grad 属性应该为 True, 这个计算梯度"
        assert temp.requires_grad == True, "temp 的 requires_grad 属性应该为 True, 这个计算梯度"

        if self.use_kl_constraint:
            loss = compute_kl_divergence(rerank_scores,cos_sim)
        elif self.use_generalized_kl_constraint:
            loss = compute_generalized_kl_divergence(rerank_scores,cos_sim)
        elif self.use_js_constraint:
            loss = compute_js_divergence(rerank_scores,cos_sim)
        else:
            rank0_print("没有使用 KL 散度、JS 散度和广义 KL 散度, 使用 f 散度当中的全变差散度")
            loss = compute_f_divergence(rerank_scores, cos_sim, tv_f_function)

        if self.use_mse_constraint:
            mse_loss = compute_mse_loss(rerank_scores,cos_sim)
            rank0_print(f"mse_loss : {mse_loss}")
            loss += 1.0*mse_loss
        if self.use_ranking_constraint:
            ranking_loss = compute_ranking_loss(rerank_scores,cos_sim)
            rank0_print(f"ranking_loss : {ranking_loss}")
            loss += 1.0*ranking_loss
           

        if self.use_feature_constraint: # 如果使用特征约束
            loss += feature_constraint_loss
        
        if self.use_distill_with_infonce:
            # 计算 infonce loss，给 pos_index_matrix 矩阵元素是 0/1 整数类型
            pos_index_matrix = torch.tensor(pos_index_list, device=cos_sim.device,requires_grad=False,dtype=torch.int)
            infonce_loss = compute_infonce_loss(cos_sim,pos_index_matrix)
            rank0_print(f"infonce_loss : {infonce_loss}")
            loss += 0.2*infonce_loss

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


def compute_infonce_loss(cos_sim, pos_index_matrix):
    """
    计算 InfoNCE 损失
    :param cos_sim: (B, N) 包含正样本和负样本的相似度矩阵
    :param pos_index_matrix: (B, N) 指示正样本位置的矩阵，正样本位置为 1，负样本位置为 0
    :return: InfoNCE 损失值
    """
    # 提取正样本的相似度
    assert cos_sim.shape == pos_index_matrix.shape, "cos_sim 和 pos_index_matrix 的形状必须相同"
    assert pos_index_matrix.dtype == torch.int, "pos_index_matrix 的数据类型必须是整数类型"
    assert pos_index_matrix.requires_grad == False, "pos_index_matrix 的 requires_grad 属性应该为 False, 这个不计算梯度"
    pos_sim = cos_sim * pos_index_matrix  # (B, N)
    # 计算正样本的对数概率
    log_prob = pos_sim - torch.logsumexp(cos_sim, dim=1, keepdim=True)  # (B, N)
    # 计算 InfoNCE 损失
    infonce_loss = -log_prob.sum(dim=1).mean()  # 平均损失
    return infonce_loss