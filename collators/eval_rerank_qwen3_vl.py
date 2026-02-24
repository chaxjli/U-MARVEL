from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen3_vision_process import process_vision_info
from .qwen3_vision_process import process_vision_info,set_max_pixels,set_video_max_pixels,set_video_total_pixels
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

class EvalRerankDataCollator(BaseDataCollator):
    def __init__(self,tokenizer, processor,is_reset_max_pixels = False,
    ):
        super().__init__(tokenizer, processor)
        # enforce left padding for decoder-only generation
        self.tokenizer.padding_side = "left"

        assert self.tokenizer.pad_token is not None, "Tokenizer must have a pad_token."
        assert self.tokenizer.padding_side == "left", "Tokenizer padding_side must be left."
        assert self.processor.tokenizer.padding_side == "left", "Processor tokenizer padding_side must be left."

        self.is_reset_max_pixels = is_reset_max_pixels
        if self.is_reset_max_pixels: 
            # set_video_max_pixels(256*28*28)
            set_video_total_pixels(16*768*28*28)
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, combine_messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        messages = []
        indexes = []

        for message, index in combine_messages:
            messages.append(message)
            indexes.append(index)
        
        assert self.tokenizer.padding_side == "left", "Tokenizer padding_side must be left."
        # rank0_print("self.tokenizer.padding_side:",self.tokenizer.padding_side)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        # processor.tokenizer.padding_side = "left" 
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        # rank0_print("inputs['input_ids'][0]:",inputs['input_ids'][0])

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None 
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        ), indexes 