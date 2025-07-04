import json
from typing import Dict, List
from torch.utils.data import Dataset
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

prompt1 = ["\nSummarize above image and sentence in one word: ","\nSummarize above sentence in one word: ","\nSummarize above image in one word: "]
prompt2 =["","",""]

class FlickrDataset(Dataset):

    def __init__(
        self, 
        image_data_path: str, 
        text_data_path: str, 
        type: str, 
        mode: str='pretrained'
    ) -> None:
        super(FlickrDataset, self).__init__()
        self.images = []
        self.image_data_path = image_data_path
        for i in range(1000):
            self.images.append(f"{i}.png")
        self.texts = json.load(open(text_data_path))
        self.type = type 
        self.mode = mode 
        self.prompt = prompt2 # 咒语为空字符串
        rank0_print("当前使用的咒语是:", self.prompt)

    def __len__(self) -> int:
        if self.type == 'image':
            return len(self.images)
        else:
            return len(self.texts)

    def construct_messages(self, text=None, image=None):
        if text is not None and image is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_data_path + '/' + image},
                        {"type": "text", "text": text},
                        {"type": "text", "text": f"{self.prompt[0]}"}
                        # {"type": "text", "text": f""}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif image is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{text}{self.prompt[1]}"}
                        # {"type": "text", "text": f""}
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
                        # {"type": "text", "text": f""}
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
        # 这里有一个需要注意的地方：图片检索任务和文本检索任务需要运行两次脚本
        # 如果是检索图片的任务：需要把 98-100 行代码注释掉，运行脚本
        # 如果是检索文本的任务：需要把 106-108 行代码注释掉，运行脚本
        if self.type == 'image':
            if self.mode == 'finetuned':
                # text = "Find an image caption describing the following everyday image."
                # text = "<instruction_start>" + text + "<instruction_end>"
                # message = self.construct_messages(image=self.images[index], text=text)
                message = self.construct_messages(image=self.images[index])
            elif self.mode == 'pretrained':
                message = self.construct_messages(image=self.images[index])
        else:
            text = self.texts[index]
            text = format_string(text)
            instruction = "Find me an everyday image that matches the given caption."
            instruction = "<instruction_start>" + instruction + "<instruction_end>"
            text = instruction + text
            message = self.construct_messages(text=text)
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class FlickrRerankI2TDataset(Dataset):

    def __init__(
        self, 
        image_data_path: str, 
        text_data_path: str, 
        ret_query_data_path: str,
        ret_cand_data_path: str,
        rank_num: int = 10
    ) -> None:
        super(FlickrRerankI2TDataset, self).__init__()
        self.images = []
        self.image_data_path = image_data_path
        for i in range(1000):
            self.images.append(f"{i}.png")
        self.texts = json.load(open(text_data_path))
        self.rank_num = rank_num 
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))

    def __len__(self) -> int:
        return len(self.images) * self.rank_num

    def construct_rerank_messages(self, query_dict, cand_dict, type='pos'):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate\
                        meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, responed with 'No'."}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidate:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'image' in cand_dict:
            cand.append({'type': 'image', 'image': cand_dict['image']})
        if 'txt' in cand_dict:
            cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def get_instance(self, index):
        instruction = "Find an image caption describing the following everyday image."
        instruction = "<instruction_start>" + instruction + "<instruction_end>"
        query_dict = {'image': self.image_data_path + '/' + self.images[index // self.rank_num], 'txt': instruction}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'txt': self.texts[cand_id]}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class FlickrRerankT2IDataset(Dataset):

    def __init__(
        self, 
        image_data_path: str, 
        text_data_path: str, 
        ret_query_data_path: str,
        ret_cand_data_path: str,
        rank_num: int = 10
    ) -> None:
        super(FlickrRerankT2IDataset, self).__init__()
        self.images = []
        self.image_data_path = image_data_path
        for i in range(1000):
            self.images.append(f"{i}.png")
        self.texts = json.load(open(text_data_path))
        self.rank_num = rank_num 
        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))

    def __len__(self) -> int:
        return len(self.texts) * self.rank_num

    def construct_rerank_messages(self, query_dict, cand_dict, type='pos'):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate\
                        meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, responed with 'No'."}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidate:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'image' in cand_dict:
            cand.append({'type': 'image', 'image': cand_dict['image']})
        if 'txt' in cand_dict:
            cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def get_instance(self, index):
        instruction = "Find me an everyday image that matches the given caption."
        instruction = "<instruction_start>" + instruction + "<instruction_end>"
        # query_dict = {'txt': f"{instruction} {self.texts[index // self.rank_num]}"}
        query_dict = {'txt': f"{instruction} {format_string(self.texts[index // self.rank_num])}"}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'image': self.image_data_path + '/' + self.images[cand_id]}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s