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