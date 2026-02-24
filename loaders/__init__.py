# 全局注册表，存储所有注册的模型加载器类，格式为 {模型名称: 加载器类}
LOADERS = {}
def register_loader(name):
    """
    类装饰器，用于将模型加载器注册到全局 LOADERS 字典
    参数: name (str): 模型唯一标识符（如 'qwen2_vl_7b'）
    返回: 装饰器函数 register_loader_cls
    """
    def register_loader_cls(cls):
        """实际执行注册的装饰器
        将模型和模型的加载器的类注册到 LOSADERS 字典当中
        返回：cls 一个模型注册类
        """
        if name in LOADERS:
            return LOADERS[name] # 名称已存在时返回已注册类（避免重复注册）
        # 注册新类
        LOADERS[name] = cls
        return cls
    return register_loader_cls

# # 导入具体模型加载器类（触发装饰器注册）
from .qwen2_vl_7b import Qwen2VL7BModelLoader       # 自动注册为某个名称
from .qwen2_vl_2b import Qwen2VL2BModelLoader       # 同上
from .qwen2_5_vl_7b import Qwen2_5_VL7BModelLoader  # 同上
from .qwen3_vl_8b import Qwen3VL8BModelLoader       # 同上
from .qwen3_vl_4b import Qwen3VL4BModelLoader       # 同上