import json
from typing import Dict, List
from torch.utils.data import Dataset
prompt1 = ["\nSummarize above image and sentence in one word:","\nSummarize above sentence in one word: ","\nSummarize above image in one word: "]
prompt2 =["","",""]
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
class CocoDataset(Dataset):

    def __init__(
        self, 
        image_data_path: str, 
        text_data_path: str, 
        type: str, 
        has_instruction = False,
        use_instruction_token = False,
    ) -> None:
        super(CocoDataset, self).__init__()
        self.images = []
        self.image_data_path = image_data_path
        for i in range(5000):
            self.images.append(f"{i}.png")
        self.texts = json.load(open(text_data_path))
        self.type = type 
        self.prompt = prompt2 # 咒语为空字符串
        rank0_print("当前使用的咒语是:", self.prompt) 

    def __len__(self) -> int:
        if self.type == 'image':
            return len(self.images)
        else:
            return len(self.texts)

    def construct_messages(self, text=None, image=None):
        if image is None:
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
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_data_path + '/' + image},
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

    def get_instance(self, index):
        if self.type == 'image':
            message = self.construct_messages(image=self.images[index])
        else:
            text = self.texts[index]
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 