from typing import Dict, Sequence

import torch
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
import copy
import os
import json
import numpy as np
import random
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
from . import register_collator
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info,set_max_pixels

@register_collator("qwen2-vl-7b")
class Qwen2VL7BDataCollator(BaseDataCollator):
    def __init__(self,tokenizer, processor,has_instruction,use_instruction_token, \
                has_hard_negative = False, has_modality_hard_negative = False,
                is_reset_max_pixels = False,has_feature_constraint=False,has_rerank_scores=False,
    ):
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

        self.is_reset_max_pixels = is_reset_max_pixels
        self.has_instruction = has_instruction                          # 数据集是否存在指令
        self.use_instruction_token = use_instruction_token              # 是否使用指令 token
        self.has_hard_negative = has_hard_negative                      # 是否使用 hard negative,不传参数的话默认使用 False
        self.has_modality_hard_negative = has_modality_hard_negative    # 是否使用 modality hard negative，不传参数的话默认使用 False
        self.has_feature_constraint = has_feature_constraint            # 是否使用 epoch1 的特征约束
        self.has_rerank_scores = has_rerank_scores                      # 是否使用 rerank 模型的 scores
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("use_instruction_token 为 True 时，has_instruction 必须为 True")
        
        if self.is_reset_max_pixels: 
            # set_max_pixels(200*28*28)  # 设置最大像素值为 200*28*28, 适配 npu, 这是 mmeb 复现 stage 1 阶段使用的最大像素值
            set_max_pixels(256*256)
            # set_max_pixels(224*224) # 20% 的数据时候显存可能会超出，所以设置为 224*224, 适配 npu

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # 来自上一阶段返回值的数量，messages: query, pos, topk_hard_neg, topk_modality_hard_neg, feature(dict), scores(dict), instruction(dict)
        # messages: [query, pos, topk_hard_neg, topk_modality_hard_neg, feature(dict), scores(dict), instruction(dict)]
        # rank0_print("len(messages) 代表 batch_size 的数量，即 query 的数量 :", len(messages))
        # rank0_print("len(messages[0]) 代表 dataset 传进来 message 的数量，:", len(messages[0]))
        batch_size = len(messages)  # batch_size 代表 query 的数量
        category_size = len(messages[0]) 
        
        # 将需要送给模型的 message 添加到 new_messages 里面
        # new_messages 是将每一个 messages 的前 new_messages_category_size 依次添加全部进去
        # 将每一个 messages 第一个都添加 new_messages 后，再添加第二个，依次类推
        new_messages = []  
        new_messages_category_size = len(messages[0])
        if self.has_instruction:           
            new_messages_category_size -=1    # 如果有指令，指令不需要添加到 new_messages 里面，所以要减去 1
        if self.has_feature_constraint:
            new_messages_category_size -=1    # 如果有特征约束，特征不需要添加到 new_messages 里面，所以要减去 1
        if self.has_rerank_scores:
            new_messages_category_size -=1    # 如果有 rerank scores，scores 不需要添加到 new_messages 里面，所以要减去 1
        for category in range(new_messages_category_size):
            for item in messages:
                new_messages.append(item[category])
        
        # rank0_print("messages :", messages)        
        feature_list = []
        scores_list = []
        for idx_fs in range(new_messages_category_size, category_size):
            for item in messages:
                if "feature" in item[idx_fs]:
                    feature_list.append(item[idx_fs]["feature"])
                if "scores" in item[idx_fs]:
                    scores_list.append(item[idx_fs]["scores"])
        # rank0_print("len(feature_list) :", len(feature_list), "len(scores_list) :", len(scores_list))

        # 处理 feature_list 格式 (query, pos, topk_hard_neg, topk_modality_hard_neg) 
        # hidden_states 的特征是 所有 query，所有 pos，所有 topk_hard_neg，所有 topk_modality_hard_neg
        if self.has_feature_constraint:
            new_feature_list = []
            assert len(feature_list[0]) == new_messages_category_size            
            for feature_idx in range(len(feature_list[0])):
                for item in feature_list:
                    new_feature_list.append(item[feature_idx])
            feature_list = copy.deepcopy(new_feature_list)
            
            if self.has_rerank_scores:
                feature_list = feature_list[:batch_size] + feature_list[2*batch_size:]
        
        
        # 处理 scores_list 格式 (topk_hard_neg, topk_modality_hard_neg)
        # 目前 scores_list 格式是 list[list] 格式, 单个元素是 list, 里面是 scores, 不需要处理
        # 需要把 new_messages 当中的 pos 取出来，因为没有单独的 pos 的 scores
        if self.has_rerank_scores:
            assert len(scores_list[0]) == new_messages_category_size - 2
            assert len(scores_list) == len(messages)
            assert len(new_messages) == new_messages_category_size * len(messages)
            
            # rank0_print("未去除正样本 len(new_messages): ", len(new_messages))
            # rank0_print("未去除正样本 new_messages: ", new_messages)
            new_messages = new_messages[:batch_size] + new_messages[2*batch_size:]
            # rank0_print("去除正样本后 len(new_messages): ", len(new_messages))
            # rank0_print("去除正样本后 new_messages: ", new_messages)

        # rank0_print("len(feature_list) :", len(feature_list), "len(scores_list) :", len(scores_list))
            

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
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
        
        
        # 索引指令的位置 -------------------------------------------------------------- debug
        if self.has_instruction and not self.use_instruction_token:
            instruction_mask = torch.zeros_like(input_ids)
            for idx, item in enumerate(messages):
                instruction = item[category_size - 1]["instruction"]
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
            instruction_end_token = "<instruction_end>"
            instruction_start_token_id = self.tokenizer.convert_tokens_to_ids(instruction_start_token)
            instruction_end_token_id = self.tokenizer.convert_tokens_to_ids(instruction_end_token)
            instruction_mask = self.get_instruction_mask(input_ids, instruction_start_token_id, instruction_end_token_id)
            # rank0_print("找到指令的位置")
        # ----------------------------------------------------------------------------- debug

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=self.has_hard_negative,                   # 是否使用 hard negative, 默认使用 False
            has_modality_hard_negative=self.has_modality_hard_negative, # 是否使用 modality hard negative, 默认使用 False
            instruction_mask=instruction_mask if self.has_instruction else None,
            feature_list=feature_list if self.has_feature_constraint else None,
            scores_list=scores_list if self.has_rerank_scores else None,
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

# #（我的版本）
# def create_fast_mask(input_ids, start_id, end_id):
#     start_mask = (input_ids == start_id).int()
#     end_mask = (input_ids == end_id).int()
#     cum_mask = (start_mask.cumsum(dim=1) - end_mask.cumsum(dim=1)) > 0
#     return (cum_mask | start_mask | end_mask).int()