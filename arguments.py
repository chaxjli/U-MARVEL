from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    # 原始参数
    model_id: str = field(default="llava-1.5-7b")
    model_local_path: Optional[str] = field(default=None)
    
    # 新添加的参数
    mean_pooling: bool = field(default=True, metadata={"help": "是否使用全局平均池化"})
    use_bi_atten: bool = field(default=True, metadata={"help": "是否使用双向注意力"})
    use_latent_atten: bool = field(default=False, metadata={"help": "是否使用潜在注意力模块"})
    use_instruction_mask: bool = field(default=True, metadata={"help": "是否使用指令掩码"})
    use_bi_loss: bool = field(default=False, metadata={"help": "是否使用双向损失"})
    use_isotropy_loss: bool = field(default=False, metadata={"help": "是否使用各向同性损失"})
    use_self_attent_pooling: bool = field(default=False, metadata={"help": "是否使用自注意力池化"})
    # 严格注意下面的参数有且仅有一个可以为True
    use_cross_entropy_loss: bool = field(default=True, metadata={"help": "是否使用交叉熵损失"})
    use_focal_infonce_loss: bool = field(default=False, metadata={"help": "是否使用FocalInfoNCELoss"})
    use_focal_infonce_abs_loss: bool = field(default=False, metadata={"help": "是否使用FocalInfoNCELossAbs"})
    use_diht_loss: bool = field(default=False, metadata={"help": "是否使用DIHTLoss"})
    use_llave_loss: bool = field(default=False, metadata={"help": "是否使用LLaVELoss"})
    use_softcse_weight_loss: bool = field(default=False, metadata={"help": "是否使用SoftCSEWeightLoss"})
    use_softcse_temperature_loss: bool = field(default=False, metadata={"help": "是否使用SoftCSETemperatureLoss"})
    
    topk_hard_negative: int = field(default=0, metadata={"help": "topk hard negative"})
    topk_modality_hard_negative: int = field(default=0, metadata={"help": "topk modality hard negative"})
    ignore_batch_other_samples: bool = field(default=False, metadata={"help": "是否忽略批次中的其他样本"})
    use_feature_constraint: bool = field(default=False, metadata={"help": "是否使用特征约束"})
    use_rerank_scores: bool = field(default=False, metadata={"help": "是否使用重排序分数"})
    use_kl_constraint: bool = field(default=True, metadata={"help": "是否使用KL约束"})
    use_js_constraint: bool = field(default=False, metadata={"help": "是否使用JS约束"})
    use_generalized_kl_constraint: bool = field(default=False, metadata={"help": "是否使用广义KL约束"})
    use_f_divergence_constraint: bool = field(default=False, metadata={"help": "是否使用f散度约束"})
    use_mse_constraint: bool = field(default=False, metadata={"help": "是否使用均方误差约束"})
    use_ranking_constraint: bool = field(default=False, metadata={"help": "是否使用排序约束"})

    use_distill_with_infonce: bool = field(default=False, metadata={"help": "是否使用infonce loss进行辅助蒸馏"})
    use_distill_with_pos: bool = field(default=False, metadata={"help": "是否使用正样本进行辅助蒸馏"})



    def __post_init__(self):
        # 原始验证逻辑,生成其他参数
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        
        # 自动填充逻辑
        if not self.model_local_path:
            self.model_local_path = self.model_hf_path
        
        # 校验损失函数参数
        self._validate_loss_functions()
    
    def _validate_loss_functions(self):
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


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data json file."})
    data_path_2: str = field(default=None, metadata={"help": "Path to the training data json file2."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation data json file."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    max_pixel: Optional[int] = field(default=None, metadata={"help": "Maximum pixel value for image normalization."})
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")
    image_data_path: str = field(default=None, metadata={"help": "Path to the image data json file."})
    text_data_path: str = field(default=None, metadata={"help": "Path to the text data json file."})
    query_data_path: str = field(default=None, metadata={"help": "Path to the query data json file."})
    cand_pool_path: str = field(default=None, metadata={"help": "Path to the cand pool data json file."})
    instructions_path: str = field(default=None, metadata={"help": "Path to the instructions data json file."})
    rerank_data_path: str = field(default=None, metadata={"help": "Path to the rerank data json file."})
    image_path_prefix: str = field(default=None, metadata={"help": "Path to the image files."})
    has_instruction: bool = field(default=False, metadata={"help": "是否进行指令匹配操作"})
    use_instruction_token: bool = field(default=False, metadata={"help": "是否使用指令 token"})
    has_hard_negative: bool = field(default=False, metadata={"help": "是否使用 hard negative"})
    has_modality_hard_negative: bool = field(default=False, metadata={"help": "是否使用 modality hard negative"})
    hard_negative_path: str = field(default=None, metadata={"help": "Path to the hard negative data json file."})
    modality_hard_negative_path: str = field(default=None, metadata={"help": "Path to the hard negative modality data json file."})
    pos_sample_path: str = field(default=None, metadata={"help": "Path to the positive sample data json file, 蒸馏rerank使用"})
    
    rerank_scores_path: str = field(default=None, metadata={"help": "Path to the rerank scores data json file, 蒸馏rerank使用"})
    query_feature_path: str = field(default=None, metadata={"help": "Path to the query feature data json file, query特征约束使用"})
    cand_feature_path: str = field(default=None, metadata={"help": "Path to the cand feature data json file, cand特征约束使用"})
    is_reset_max_pixels: bool = field(default=False, metadata={"help": "是否重置最大像素值"})
    has_distill_with_pos: bool = field(default=False, metadata={"help": "是否使用正样本进行辅助蒸馏"})
    

    def __post_init__(self):
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("use_instruction_token requires has_instruction to be True")
        if self.has_distill_with_pos:
            assert self.rerank_scores_path is not None, "has_distill_with_pos requires rerank_scores_path to be set"
            assert self.pos_sample_path is not None, "has_distill_with_pos requires pos_sample_path to be set"
            assert self.hard_negative_path is not None, "has_distill_with_pos requires hard_negative_path to be set"
            assert self.modality_hard_negative_path is not None, "has_distill_with_pos requires modality_hard_negative_path to be set"
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = False
    train_vision_encoder: bool = False
    train_vision_projector: bool = False
    vision_projector_lr: float = None
    train_temperature: bool = field(default=False, metadata={"help": "是否训练温度参数"})
 

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = True
    use_vision_lora: bool = True
    q_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lora_r: int = 16
    vision_lora_alpha: int = 16
    task_type: str = "TOKEN_CLS"