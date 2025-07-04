import json
from typing import Dict, List
from torch.utils.data import Dataset

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
        if self.type == 'image':
            if self.mode == 'finetuned':
                text = "Find an image caption describing the following everyday image."
                message = self.construct_messages(image=self.images[index], text=text)
            elif self.mode == 'pretrained':
                message = self.construct_messages(image=self.images[index])
        else:
            text = self.texts[index]
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
        query_dict = {'txt': f"{instruction} {self.texts[index // self.rank_num]}"}
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        cand_dict = {'image': self.image_data_path + '/' + self.images[cand_id]}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 