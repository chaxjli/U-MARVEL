from typing import Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from . import register_loader      # 自定义装饰器，用于注册模型加载器
from .base import BaseModelLoader  # 基础加载器类
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration                   # 预训练模型结构
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration  # 微调模型结构

# 注册模型加载器到全局注册表，"qwen2-vl-7b" 为模型标识符
@register_loader("qwen2-vl-7b")
class Qwen2VL7BModelLoader(BaseModelLoader):
    """
    Qwen2-VL-7B 多模态模型加载器
    核心功能：加载预训练/微调模型， 处理多模态输入（图像+文本）， 动态扩展词表
    """
    
    def load(self, load_model: bool = True, pretrain=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        """
        加载模型的三件套（模型、分词器、处理器）
        参数说明：
            load_model: 是否加载模型权重（可用于仅加载分词器）
            pretrain: True - 加载预训练模型，False - 加载微调模型
        """
        # 模型加载分支 =================================================================
        if load_model and pretrain:
            # 加载预训练模型（通常用于继续预训练或特征提取）
            model = Qwen2VLRetForConditionalGeneration.from_pretrained(
                self.model_local_path,   # 本地模型路径
                **self.loading_kwargs,   # 加载参数（包含 device_map, quantization_config 等）
            )
        elif load_model and not pretrain:
            # 加载微调模型（通常用于下游任务推理）
            model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
                self.model_local_path,
                strict=False,  # 宽松加载，允许加载部分权重   
                **self.loading_kwargs,
            )

        # 处理器与分词器加载 ==========================================================
        processor = AutoProcessor.from_pretrained(self.model_local_path) # 多模态处理器（处理图像+文本）
        tokenizer = processor.tokenizer                                  # 从处理器中提取文本分词器
        
        # 词表扩展 ===================================================================
        self.add_embed_token(tokenizer, model)
        return model, tokenizer, processor 

    def add_embed_token(self, tokenizer, model, emb_token="<emb>"):
        """
        功能: 动态添加特殊嵌入标记
        处理: 扩展词表后需要同步调整模型 embedding 层大小，确保新增 token 的嵌入初始化合理，配置文件中记录新增 token 的ID
        """
        
        # 添加新 token 到分词器
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        
        # 校验是否成功添加（防止重复添加导致计数错误）
        assert len(emb_tokens) == num_new_tokens, \
            f"Failed to add tokens: expected {len(emb_tokens)}, got {num_new_tokens}"

        # 调整模型 embedding 层大小（关键操作）
        # 注意：此处可能影响 decoder 的输出层，需要确认模型是否自动同步调整
        model.resize_token_embeddings(len(tokenizer))

        # 获取新增 token 的 ID 并保存到配置
        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids  # 用于后续识别这些特殊token
        
        """
        初始化策略说明：
        - 新增 token 的 embedding 默认使用随机初始化
        - 可在后续通过 model.init_token_embeddings() 自定义初始化
        - 对于多模态模型，可能需要特殊处理视觉-文本对齐
        """


