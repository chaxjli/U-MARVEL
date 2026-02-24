# 抽象基类（ABC）用于定义模型加载器的统一接口
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoProcessor, BitsAndBytesConfig

class BaseModelLoader(ABC):
    def __init__(
        self, 
        model_hf_path: str,                              # Hugging Face 仓库模型标识
        model_local_path: str,                           # 本地模型目录路径
        compute_dtype: torch.dtype,                      # 计算数据类型（如 torch.float16）
        bnb_config: Optional[BitsAndBytesConfig] = None, # 量化配置（4/8-bit）
        use_flash_attn: bool = False,                    # 是否启用Flash Attention加速
        device_map: Optional[Union[Dict, str]] = None,   # 设备映射策略
    ) -> None:
        # 初始化模型路径参数
        self.model_hf_path = model_hf_path
        self.model_local_path = model_local_path
        
        # 构建模型加载的关键参数字典
        self.loading_kwargs = dict(
            torch_dtype=compute_dtype,         # 控制模型权重数据类型
            quantization_config=bnb_config,    # 量化配置（QLoRA等场景）
            device_map=device_map,             # 分布式设备映射策略
            # 示例: device_map="auto" 表示自动分配多GPU/NPU设备
        )
        
        # Flash Attention 加速配置（硬件和模型兼容性要求高）
        if use_flash_attn: # 需要安装 flash-attn 库且硬件支持
            self.loading_kwargs["attn_implementation"] = "flash_attention_2"

    @abstractmethod # 强制子类实现 load 方法
    def load(self, load_model: bool = True) -> Tuple[
        PreTrainedModel, 
        Union[None, PreTrainedTokenizer], 
        Union[None, AutoProcessor]
    ]: 
        """
        参数说明：
            load_model - 是否实际加载模型权重（可用于仅加载分词器/处理器）
        返回值三元组：
            PreTrainedModel: 加载的模型实例
            PreTrainedTokenizer: 文本分词器（多模态模型可能为None）
            AutoProcessor: 多模态处理器（纯文本模型可能为None）
        """
        ...