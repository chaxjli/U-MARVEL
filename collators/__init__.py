COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

from .qwen2_vl_7b import Qwen2VL7BDataCollator
from .qwen2_vl_2b import Qwen2VL2BDataCollator
from .qwen2_5_vl_7b import Qwen2_5_VL7BDataCollator
from .qwen3_vl_8b import Qwen3VL8BDataCollator
from .qwen3_vl_4b import Qwen3VL4BDataCollator