# 导入类型注解和必要的库
from typing import Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from .base import BaseModelLoader  # 从当前包导入基础模型加载器

class Qwen2VLModelLoader(BaseModelLoader):
    """用于加载 Qwen2VL 多模态生成模型的专用加载器
    
    特性：
    - 支持条件生成任务的视觉语言模型加载
    - 自动处理多模态输入处理器
    - 继承自基础模型加载器，复用通用配置
    """
    
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        """
        加载模型及其关联组件
        
        参数:
        load_model (bool): 是否实际加载模型权重，默认为True。
                           设为False时仅加载配置（当前实现需调整）
        
        返回:
        Tuple[AutoModelForCausalLM, AutoTokenizer, None]: 
            - 模型实例 (None当load_model=False)
            - 分词器实例
            - 多模态处理器 (实际返回processor，与类型标注不符需要修正)
        
        异常:
        FileNotFoundError: 当模型路径不存在时抛出
        ValueError: 模型配置错误时抛出
        """
        # 模型加载分支
        if load_model:
            # 从本地路径加载预训练模型，自动识别模型架构
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_local_path,  # 继承自基类的模型存储路径
                ​**self.loading_kwargs,  # 继承的加载参数（如torch_dtype, device_map等）
            )
        else:
            # 当前实现存在缺陷：当load_model=False时返回未定义的model变量
            # 建议修改为：model = None
            pass  # 需要添加处理逻辑

        # 加载多模态处理器（自动处理图像+文本输入）
        # 能自动识别处理以下输入类型：
        # - 图像（PNG/JPG等）
        # - 自然语言文本
        # - 可能的其他模态数据
        processor = AutoProcessor.from_pretrained(self.model_local_path)

        # 从处理器中分离文本分词器
        # 注意：这里的分词器可能包含特殊的多模态token（如<image>标记）
        tokenizer = processor.tokenizer

        # 返回元组（注意类型标注与实际返回值的processor不符）
        # 建议修正类型标注为：-> Tuple[Optional[AutoModel], AutoTokenizer, AutoProcessor]
        return model, tokenizer, processor  # 实际返回processor，但类型标注为None，需要修正

        # 典型使用示例：
        # model, tokenizer, processor = Qwen2VLModelLoader("/path/to/model").load()
        # inputs = processor(text="描述这张图片", images=image, return_tensors="pt")
        # outputs = model.generate(**inputs)