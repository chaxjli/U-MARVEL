from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info
from .qwen2_vision_process import process_vision_info,set_max_pixels,set_video_max_pixels,set_video_total_pixels


class EvalRerankDataCollator(BaseDataCollator):
    def __init__(self,tokenizer, processor,is_reset_max_pixels = False,
    ):
        super().__init__(tokenizer, processor)

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

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

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