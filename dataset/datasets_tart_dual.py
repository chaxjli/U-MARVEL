import os
import json
from torch.utils.data import Dataset
from typing import Dict, List # debug
import random 
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler,rank0_print_nested_dict,
    read_json_file,
)
prompt1 = ["\nSummarize above image and sentence in one word: ","\nSummarize above sentence in one word: ","\nSummarize above image in one word: "]
prompt2 =["","",""]
class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self, 
        data_path: str, 
        tokenizer = None,
        has_instruction=True,
        use_instruction_token = False,
        has_hard_negative=False,
        has_modality_hard_negative=False, 
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        
        # 从文件中加载数据--------------------------------------------------------------
        self.all_data = read_json_file(data_path)
        
        # 进行处理的一些参数--------------------------------------------------------------
        self.tokenizer = tokenizer         
        self.has_instruction = has_instruction   # MBEIR 数据集中存在指令
        self.prompt = prompt2                    # 咒语为空字符串
        self.use_instruction_token = use_instruction_token  # 是否使用指令 token
        self.has_hard_negative = has_hard_negative  # 是否使用 hard negative
        self.has_modality_hard_negative = has_modality_hard_negative  # 是否使用 modality hard negative
        
        rank0_print("这是 LazySupervisedDataset 的初始化函数 打印的信息，-----------------------------")
        rank0_print("当前使用的咒语是: ", self.prompt)
        rank0_print("当前数据集是否存在指令: ", self.has_instruction)
        rank0_print("当前是否使用指令 token: ", self.use_instruction_token)
        rank0_print("当前是否使用 hard negative: ", self.has_hard_negative)
        rank0_print("当前是否使用 modality hard negative: ", self.has_modality_hard_negative)

        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("Instruction token is enabled but the dataset does not have instructions.")


    def __len__(self) -> int:
        return len(self.all_data)

    # 构建消息，传进来一个数据字典 data_dict，返回一个消息列表
    # 这个 data_dict 是一个字典，是由 get_instance 方法返回的
    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{data_dict['txt']}{self.prompt[0]}"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'txt' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{data_dict['txt']}{self.prompt[1]}"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{self.prompt[2]}"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        return message

# 这是 all_data 中的一个实例-------------------------------------------------------------
# {'instruction': 'Find an evidence paragraph on the web with evidence and answer for this',
#  'query_text': 'What happens when you break a bone',
#  'pos_text': 'A fracture is a break, usually in a bone. If the broken bone punctures the skin, it is called an open or compound fracture. Fractures commonly happen because of car accidents, falls or sports injuries.',
#  'neg_text': 'Marion, Iowa. Marion is a city in Linn County, Iowa, United States. The population was 26,294 at the 2000 census and was 34,768 at the 2010 census, an increase of 32.2%. The city is located next to Cedar Rapids and part of the Cedar Rapids Metropolitan Statistical Area.',
#  'hard_neg_text': 'Marion, Iowa. Marion is a city in Linn County, Iowa, United States. The population was 26,294 at the 2000 census and was 34,768 at the 2010 census, an increase of 32.2%. The city is located next to Cedar Rapids and part of the Cedar Rapids Metropolitan Statistical Area.'}
    
    # 获取实例，传进来一个索引，返回一个 instance 字典
    def get_instance(self, index):
        data_entry = self.all_data[index]
        query_txt = data_entry.get('query_text') or ""
        query_prompt = data_entry.get('instruction') or ""

        pos_cand_txt = data_entry.get('pos_text') or ""
        pos_cand_txt = format_string(pos_cand_txt)
        neg_cand_txt = data_entry.get('neg_text') or ""
        neg_cand_txt = format_string(neg_cand_txt)
        modality_hard_neg_cand_txt = data_entry.get('hard_neg_text') or ""
        modality_hard_neg_cand_txt = format_string(modality_hard_neg_cand_txt)

        # rank0_print("data_entry: ", data_entry)
        # rank0_print("query_txt: ", query_txt)
        # rank0_print("query_prompt: ", query_prompt)
        # rank0_print("pos_cand_txt: ", pos_cand_txt)
        # rank0_print("neg_cand_txt: ", neg_cand_txt) 
        # rank0_print("modality_hard_neg_cand_txt: ", modality_hard_neg_cand_txt)

        # 添加指令特殊 token -----------------------------------------------------------------
        if self.use_instruction_token:
            query_prompt = "<instruction_start>" + query_prompt + "<instruction_end>"
        # -----------------------------------------------------------------------------------
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(f"{query_txt}")

        # truncation processing is applied to prevent memory overflow.
        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])
        if self.has_hard_negative:
            neg_cand_txt = self.tokenizer(neg_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
            neg_cand_txt = self.tokenizer.decode(neg_cand_txt['input_ids'])
        if self.has_modality_hard_negative:
            modality_hard_neg_cand_txt = self.tokenizer(modality_hard_neg_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
            modality_hard_neg_cand_txt = self.tokenizer.decode(modality_hard_neg_cand_txt['input_ids'])
                
        query = {"txt": query_txt_with_prompt}
        pos_cand = {"txt": pos_cand_txt}

        if self.has_hard_negative:
            neg_cand = {"txt": neg_cand_txt}

        if self.has_modality_hard_negative:
            modality_hard_neg_cand = {"txt": modality_hard_neg_cand_txt}

        instance = {"query": query}
        instance.update({"pos_cand": pos_cand})

        # 添加 hard negative 和 modality hard negative 和 指令mask---------------------------------
        if self.has_hard_negative:
            instance.update({"neg_cand": neg_cand})
        if self.has_modality_hard_negative:
            instance.update({"modality_hard_neg_cand": modality_hard_neg_cand})
        instance.update({"instruction": query_prompt})
        # -----------------------------------------------------------------------------

        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']

        neg_cand_dict = instance.get('neg_cand', None)
        modality_hard_neg_cand_dict = instance.get('modality_hard_neg_cand', None)
        
        # 获取指令信息, 处理成字典格式
        instruction_message = dict()
        instruction = instance['instruction']
        instruction_message.update({"instruction": instruction})

        query_message = self.construct_messages(query_dict)
        cand_message = self.construct_messages(cand_dict)
        neg_cand_message = self.construct_messages(neg_cand_dict) if neg_cand_dict else None
        modality_hard_neg_cand_message = self.construct_messages(modality_hard_neg_cand_dict) if modality_hard_neg_cand_dict else None

        result_list = [query_message, cand_message, neg_cand_message, modality_hard_neg_cand_message]
        result_list = [item for item in result_list if item is not None]
        if self.has_instruction:
            result_list.append(instruction_message)
        result_tuple = tuple(result_list)
        return result_tuple

    # 自定义的 select 方法，用于截取数据集的一部分---------------------------------------------
    def select(self, indices):
        import copy
        """安全创建子数据集 (绕过文件路径校验)"""
        # 创建空壳对象
        new_dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
        
        # 手动初始化必要属性
        new_dataset.all_data = [copy.deepcopy(self.all_data[i]) for i in indices]     
        new_dataset.tokenizer = self.tokenizer
        new_dataset.has_instruction = self.has_instruction
        new_dataset.has_hard_negative = self.has_hard_negative
        new_dataset.has_modality_hard_negative = self.has_modality_hard_negative
        new_dataset.prompt = copy.deepcopy(self.prompt)
        new_dataset.use_instruction_token = self.use_instruction_token
        
        # 跳过非必要初始化步骤
        if hasattr(self, '_is_initialized'):
            new_dataset._is_initialized = True
        
        return new_dataset
    #-------------------------------------------------------------------------------------


def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s