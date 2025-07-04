from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

class MbeirQueryDataCollator(BaseDataCollator):
    def __init__(self,tokenizer, processor,has_instruction,use_instruction_token):
        """
        Args:
            tokenizer: PreTrainedTokenizer
            processor: AutoProcessor
            has_instruction: bool, 数据集是否存在指令
        remark: 这个类仅训练模型使用，train_nli 和 train_mbier 都会用到
        Returns:
            Dict[str, torch.Tensor]
        """
        super().__init__(tokenizer, processor)
        self.has_instruction = has_instruction # 是否使用 instruction
        self.use_instruction_token = use_instruction_token # 是否使用指令 token
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("use_instruction_token 为 True 时，has_instruction 必须为 True")
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        qids = []
        # 打印一下新的消息
        # rank0_print("messages[0]: ", messages[0])
        for item in messages: # item: [query_message, qid, (instruction_message)]
            new_messages.append(item[0])
            qids.append(item[1])
        
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
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
        
        has_hard_negative = False
        # 索引指令的位置 -------------------------------------------------------------- debug
        if self.has_instruction and not self.use_instruction_token:
            instruction_mask = torch.zeros_like(input_ids)
            for idx, item in enumerate(messages):
                instruction = item[2]["instruction"]
                instruction_tokens = self.tokenizer(instruction, truncation=True, max_length=480, padding=False, return_tensors="pt", add_special_tokens=False)
                instruction_tokens_ids = instruction_tokens['input_ids'][0]  # Shape: (instruction_len,)
                
                # 使用 unfold 进行向量化匹配
                instruction_len = instruction_tokens_ids.size(0)
                input_ids_len = input_ids[idx].size(0)
                
                if input_ids_len < instruction_len:
                    instruction_start = None
                else:
                    # 滑动窗口展开 (效率关键点)
                    windows = input_ids[idx].unfold(0, instruction_len, 1)  # Shape: (num_windows, instruction_len)
                    matches = (windows == instruction_tokens_ids).all(dim=1)  # 向量化比较
                    match_indices = matches.nonzero(as_tuple=True)[0]
                    if match_indices.numel() > 0:
                        instruction_start = match_indices[0].item()
                    else:
                        instruction_start = None
                if instruction_start is not None:
                    instruction_end = instruction_start + instruction_len
                    instruction_mask[idx][instruction_start:instruction_end] = 1
                    # rank0_print("找到指令的位置")
                else:
                    # 调试信息保持不变
                    instruction_mask = None
                    rank0_print("input_ids:", input_ids[idx])
                    rank0_print("分词后的指令 token IDs:", instruction_tokens_ids)
                    input_text = self.tokenizer.decode(input_ids[idx][:])
                    rank0_print("input_ids 对应的文本:", input_text)
                    rank0_print("原始指令文本:", instruction)
                    raise ValueError("未找到指令的位置")
        elif self.has_instruction and self.use_instruction_token:
            instruction_start_token = "<instruction_start>"
            instruction_end_token   = "<instruction_end>"
            instruction_start_token_id = self.tokenizer.convert_tokens_to_ids(instruction_start_token)
            instruction_end_token_id = self.tokenizer.convert_tokens_to_ids(instruction_end_token)
            instruction_mask = self.get_instruction_mask(input_ids, instruction_start_token_id, instruction_end_token_id)
        # ----------------------------------------------------------------------------- debug 
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            qids=qids,
            instruction_mask=instruction_mask if self.has_instruction else None,
        )
    # 元宝优化的版本
    def get_instruction_mask(self, input_ids, start_id, end_id):
        # 处理输入为空张量的情况
        if input_ids.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.int)

        start_mask = (input_ids == start_id).int()
        end_mask = (input_ids == end_id).int()

        # 累积求和
        start_cum = start_mask.cumsum(dim=1)
        end_cum = end_mask.cumsum(dim=1)

        # 确保开始标记在结束标记之前，避免标记顺序混乱的问题
        valid_mask = start_cum >= end_cum
        cum_mask = (start_cum - end_cum) > 0

        # 处理嵌套情况：通过限制每个开始标记只能对应一个结束标记
        valid_start = start_mask * (valid_mask & (start_cum - end_cum == 1))
        valid_end = end_mask * (valid_mask & (start_cum - end_cum == 0))

        # 生成最终的掩码矩阵
        instruction_mask = (cum_mask | valid_start | valid_end).int()
        return instruction_mask

class MbeirCandidateDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        dids = []

        for item in messages:
            new_messages.append(item[0])
            dids.append(item[1])

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
        # inputs = self.processor(
        #     text=texts,
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding='longest',
            truncation=True,
            max_length=1024,
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
        
        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            dids=dids
        )