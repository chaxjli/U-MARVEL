from typing import Tuple

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from . import register_loader
from .base import BaseModelLoader
# from models.qwen3_vl import Qwen3VLRetForConditionalGeneration
from models.qwen3_vl_finetune import Qwen3VLRetFinetuneForConditionalGeneration
from models.qwen3_vl_finetune_with_hard_neg import Qwen3VLRetFinetuneHardNegForConditionalGeneration
from models.qwen3_vl_finetune_with_distillrerank import Qwen3VLRetFinetuneDistillRerankForConditionalGeneration
@register_loader("qwen3-vl-4b")
class Qwen3VL4BModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True, pretrain=True,hard_neg = False,distillrerank = False,use_bi_atten = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if distillrerank:
            model = Qwen3VLRetFinetuneDistillRerankForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                use_bi_atten=use_bi_atten,
                **self.loading_kwargs,
            )
        else:
            if load_model and pretrain and not hard_neg:
                model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    use_bi_atten=use_bi_atten,
                    **self.loading_kwargs,
                ) 
            elif load_model and not pretrain and not hard_neg:
                model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    use_bi_atten=use_bi_atten,
                    **self.loading_kwargs,
                )
            elif load_model and not pretrain and hard_neg:
                model = Qwen3VLRetFinetuneHardNegForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    use_bi_atten=use_bi_atten,
                    **self.loading_kwargs,
                ) 

        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        # Ensure left padding for decoder-only generation
        tokenizer.padding_side = "left"
        assert tokenizer.pad_token is not None, "Tokenizer must have a pad_token."

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