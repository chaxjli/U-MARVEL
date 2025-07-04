from typing import Dict, List
from torch.utils.data import Dataset
import pandas as pd 
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
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
        # tokenizer = None, 
        has_instruction=False,
        use_instruction_token = False,
        has_hard_negative=False,
        has_modality_hard_negative=False,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.csv = pd.read_csv(data_path)
        self.prompt = prompt2 # 咒语为空字符串
        self.has_instruction = has_instruction # NLI 数据集中不存在指令
        self.use_instruction_token = use_instruction_token
        self.has_hard_negative = has_hard_negative
        self.has_modality_hard_negative = has_modality_hard_negative
        rank0_print("当前使用的咒语是:", self.prompt)
        rank0_print("数据集中是否存在指令:", self.has_instruction)
    def __len__(self) -> int:
        return len(self.csv) 

    def construct_messages(self, text):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}{self.prompt[1]}"}
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

    def get_instance(self, index):
        sent0, sent1, sent2 = self.csv['sent0'][index], self.csv['sent1'][index], self.csv['hard_neg'][index]
        message1 = self.construct_messages(sent0)
        message2 = self.construct_messages(sent1)
        message3 = self.construct_messages(sent2)

        # return message1, message2, message3 
        return message1, message2 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i)
    
    # 自定义的 select 方法，用于截取数据集的一部分---------------------------------------------
    def select(self, indices):
        import copy
        """安全创建子数据集 (绕过文件路径校验)"""
        # 创建空壳对象
        new_dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
        
        # 手动初始化必要属性,一行一行复制， debug
        new_dataset.csv = self.csv.iloc[indices].copy()   
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