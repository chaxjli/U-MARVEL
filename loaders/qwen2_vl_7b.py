from typing import Tuple

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from . import register_loader
from .base import BaseModelLoader
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
from models.qwen2_vl_finetune_with_hard_neg import Qwen2VLRetFinetuneHardNegForConditionalGeneration
from models.qwen2_vl_finetune_with_distillrerank import Qwen2VLRetFinetuneDistillRerankForConditionalGeneration
@register_loader("qwen2-vl-7b")
class Qwen2VL7BModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True, pretrain=True,hard_neg = False,distillrerank = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if distillrerank:
            model = Qwen2VLRetFinetuneDistillRerankForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
        else:
            if load_model and pretrain and not hard_neg:
                model = Qwen2VLRetForConditionalGeneration.from_pretrained(
                    self.model_local_path, 
                    **self.loading_kwargs,
                ) 
            elif load_model and not pretrain and not hard_neg:
                model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                )
            elif load_model and not pretrain and hard_neg:
                model = Qwen2VLRetFinetuneHardNegForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                ) 

        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        self.add_embed_token(tokenizer, model, emb_token="<emb>")  # add emb token 获取嵌入 mask 的位置
        # add instruction tokens 获取指令 mask 的位置
        self.add_embed_token(tokenizer, model, emb_token="<instruction_start>")
        self.add_embed_token(tokenizer, model, emb_token="<instruction_end>")

        return model, tokenizer, processor 

    def add_embed_token(self, tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens

        model.resize_token_embeddings(len(tokenizer))

        # 为每个 token 创建独立配置项
        token_id = tokenizer.convert_tokens_to_ids(emb_token)
        
        # 根据 token 类型动态设置 config
        if emb_token == "<instruction_start>":
            model.config.instruction_start_token_id = token_id
        elif emb_token == "<instruction_end>":
            model.config.instruction_end_token_id = token_id
        else:
            model.config.emb_token_id = token_id  # 默认通用 token