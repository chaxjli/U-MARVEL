from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    """存储模型相关的配置参数"""
    
    # 模型名称或路径，用于指定Hugging Face模型
    model_name: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    
    # 模型骨干网络名称，可选参数
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    
    # 处理器名称，用于处理输入数据
    processor_name: str = field(
        default=None, metadata={"help": "processor_name, huggingface model name or path"}
    )
    
    # LAVIS框架中的模型类型
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    
    # 本地模型检查点路径
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    
    # 编码器的池化方法，默认为最后一层
    pooling: str = field(
        default='last',
        metadata={"help": "pooling method for encoder"}
    )
    
    # 是否对查询和段落表示进行归一化
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )
    
    # softmax温度参数，控制分布的平滑度
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    
    # 是否使用LoRA进行参数高效微调
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    
    # LoRA的秩参数，控制低秩矩阵的维度
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    
    # LoRA的缩放参数
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    
    # LoRA的dropout率
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    
    # LoRA目标模块名称，用逗号分隔
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    
    # 图像编码器中使用的裁剪数量
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )


@dataclass
class DataArguments:
    """存储数据处理相关的配置参数"""
    
    # Hugging Face数据集名称
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    
    # 数据集分割名称，支持原始分割或多样化指令分割
    split_name: List[str] = field(
        default='original', metadata={"help": "'original', 'diverse_instruction'"}
    )
    
    # 数据集子集名称，适用于有多个子集的数据集
    subset_name: List[str] = field(
        default=None, metadata={"help": "Useful for datasets with subsets"}
    )
    
    # 数据集分割类型，如训练集、验证集等
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    
    # 每个子集的训练样本数量
    num_sample_per_subset: int = field(
        default=100, metadata={"help": "number of training samples per subset"}
    )
    
    # 图像目录路径
    image_dir: str = field(
        default=None, metadata={"help": "Image directory path"}
    )
    
    # 编码输出路径
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )
    
    # 输入序列的最大长度，注意可能会截断文本提示
    max_len: int = field(
        default=None, metadata={"help": "The maximum total input sequence length after tokenization. "
                                        "Use with caution, since it may truncate text prompts due to large image lengths."},
    )
    
    # 嵌入类型
    embedding_type: str = field(
        default="", metadata={"help": "embedding type"}
    )
    
    # 图像分辨率，适用于某些模型的图像预处理
    image_resolution: str = field(
        default='high', metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """继承自Hugging Face的TrainingArguments，添加自定义训练参数"""
    
    # 是否冻结图像编码器
    image_encoder_freeze: bool = field(
        default=False, metadata={"help": "huggingface model name"}
    )
    
    # 训练模型的保存目录
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    
    # 项目名称，用于日志和监控
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )

    # 日志记录步数间隔
    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    
    # 训练轮数
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    
    # 是否使用梯度缓存更新
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"})
    
    # 查询端子集大小，用于梯度缓存
    gc_q_chunk_size: int = field(
        default=2, metadata={"help": "query side subset size"})
    
    # 目标端子集大小，用于梯度缓存
    gc_p_chunk_size: int = field(
        default=2, metadata={"help": "target side subset size"})


@dataclass
class MTEBArguments:
    """存储MTEB基准测试相关的配置参数"""
    
    # 任务类型列表
    task_types: List[str] = field(
        default=None, metadata={"help": ""}
    )
    
    # 具体任务列表
    tasks: List[str] = field(
        default=None, metadata={"help": ""}
    )