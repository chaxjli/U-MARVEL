from typing import Tuple

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from .base import BaseModelLoader


class Qwen3VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            ) 

        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        # Ensure left padding for decoder-only generation
        tokenizer.padding_side = "left"
        assert tokenizer.pad_token is not None, "Tokenizer must have a pad_token."

        return model, tokenizer, processor 