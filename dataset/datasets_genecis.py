import json
from typing import Dict, List
from torch.utils.data import Dataset
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
prompt1 = ["\nSummarize above image and sentence in one word: ","\nSummarize above sentence in one word: ","\nSummarize above image in one word: "]
prompt2 =["","",""]
# 第一版对应的结果是 0.130
# instruction1 = "I'm looking for a similar everyday image with the described changes." # 第一个
# instruction2 = "I'm looking for an image with the same attributes as described."      # 第二个

# 第二版对应的结果是 0.180
instruction1 = "I want you to retrieve an similar image with the described changes. Show me an image that best captures the following the described changes."
instruction2 = "I want you to retrieve an image with the same attributes as described. Show me an image that best captures the following the same attributes."

# Find me an everyday image that matches the given caption.
# Based on the following fashion description, retrieve the best matching image.	
# Match the provided description to the correct fashion item photo.	
# Identify the fashion image that aligns with the described product.	
# You need to identify the image that corresponds to the fashion product description provided.

# 第三版对应的结果是 0.183
# instruction1 = "I'm looking for a similar everyday image with the described changes. I want you to retrieve an similar image with the described changes. Show me an image that best captures the following the described changes."
# instruction2 = "I'm looking for an image with the same attributes as described. I want you to retrieve an image with the same attributes as described. Show me an image that best captures the following the same attributes."

# 第四版对应的结果是 0.177
# instruction1 = "I'm in search of an everyday image that closely resembles another one, with specific changes as described. Please find and retrieve an image that precisely matches the described modifications and has a similar context to an ordinary daily scene. Display the image that most effectively captures these described changes in a way that closely aligns with everyday life."
# instruction2 = "I'm seeking an everyday image that has identical attributes to the ones I've described. Kindly search for and retrieve an image that perfectly matches the specified attributes and represents a typical daily situation. Show me the image that best encapsulates these same attributes in an everyday context, ensuring a high degree of accuracy and similarity."

# 第五版对应的结果是 0.157
# instruction1 = "I'm seeking an everyday image. It should closely mirror another image in terms of scene layout and general content. The described changes, like altering an object's color, adding or removing a specific item, or adjusting the light intensity, must be precisely reflected in the retrieved image. Ensure the image represents a common daily life situation. Please find and show me the image that most accurately matches these described changes. I'm looking for a similar everyday image with the described changes. I want you to retrieve an similar image with the described changes. Show me an image that best captures the following the described changes."
# instruction2 = "I require an everyday image with attributes that precisely match the description. Attributes cover object shapes, sizes, colors, and the overall scene environment. The image should depict a typical daily moment. Retrieve an image where every specified attribute aligns perfectly, guaranteeing a high level of accuracy in the match. I'm looking for an image with the same attributes as described. I want you to retrieve an image with the same attributes as described. Show me an image that best captures the following the same attributes."

class GenecisCOCODataset(Dataset):

    def __init__(
        self, 
        annotation_path,
        image_path_prefix,
        tokenizer,
        type: str="query"
    ) -> None:

        super(GenecisCOCODataset, self).__init__()
        self.type = type
        self.tokenizer = tokenizer 
        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for item in self.val_samples:
            self.gallery_ids.add(str(item['target']['val_image_id']))
            gallery = item['gallery']
            for x in gallery:
                self.gallery_ids.add(str(x['val_image_id']))

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix
        self.prompt = prompt2 # 咒语为空字符串
        rank0_print("当前使用的咒语是:", self.prompt)
    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.val_samples)
        elif self.type == 'image':
            return len(self.gallery_ids)

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
        elif text is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
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
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{text}"},
                        {"type": "text", "text": f"{self.prompt[0]}"}
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
        if self.type == 'query':
            instruction = instruction1
            instruction = "<instruction_start>" + instruction + "<instruction_end>"
            sample = self.val_samples[index]
            reference_name = str(sample['reference']['val_image_id'])
            reference_img_path = f"{self.image_path_prefix}/{reference_name.zfill(12)}.jpg"
            relative_caption = sample['condition']
            relative_caption = format_string(relative_caption)
            # relative_caption = self.tokenizer(relative_caption, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
            # relative_caption = self.tokenizer.decode(relative_caption['input_ids'])
            relative_caption = f"{instruction} {relative_caption}"
            query_message = self.construct_messages(text=relative_caption, image=reference_img_path)
            return query_message
        elif self.type == 'image':
            image_id = self.gallery_ids[index]
            image = f"{self.image_path_prefix}/{image_id.zfill(12)}.jpg"
            candidate_message = self.construct_messages(image=image)
            return candidate_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 



class GenecisVAWDataset(Dataset):

    def __init__(
        self, 
        annotation_path,
        image_path_prefix,
        tokenizer,
        type: str="query",
    ) -> None:

        super(GenecisVAWDataset, self).__init__()
        self.type = type 
        self.tokenizer = tokenizer

        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for index, item in enumerate(self.val_samples):
            self.gallery_ids.add(f"{str(item['target']['image_id'])}_{index}_1.jpg")
            gallery = item['gallery']
            for i, x in enumerate(gallery):
                self.gallery_ids.add(f"{str(x['image_id'])}_{index}_{2 + i}.jpg")

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix
        self.prompt = prompt2 # 咒语为空字符串
        rank0_print("当前使用的咒语是:", self.prompt)
    def __len__(self) -> int:
        if self.type == 'query':
            return len(self.val_samples)
        elif self.type == 'image':
            return len(self.gallery_ids)

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
        elif text is None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
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
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{text}"},
                        {"type": "text", "text": f"{self.prompt[0]}"}
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
        if self.type == 'query':
            if 'change_attribute' in self.image_path_prefix:
                instruction = instruction1
                instruction = "<instruction_start>" + instruction + "<instruction_end>"
            elif 'focus_attribute' in self.image_path_prefix:
                instruction = instruction2
                instruction = "<instruction_start>" + instruction + "<instruction_end>"
            sample = self.val_samples[index]
            reference_name = str(sample['reference']['image_id'])
            reference_img_path = f"{self.image_path_prefix}/{reference_name}_{index}_0.jpg"
            relative_caption = sample['condition']
            relative_caption = format_string(relative_caption)
            # relative_caption = self.tokenizer(relative_caption, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
            # relative_caption = self.tokenizer.decode(relative_caption['input_ids'])
            relative_caption = f"{instruction} {relative_caption}"
            query_message = self.construct_messages(text=relative_caption, image=reference_img_path)
            return query_message
        elif self.type == 'image':
            image_id = self.gallery_ids[index]
            image = f"{self.image_path_prefix}/{image_id}"
            candidate_message = self.construct_messages(image=image)
            return candidate_message


    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 

# 下面是 rerank 模型使用的 dataset --------------------------------------------------------------------------------------------------------

class GenecisCOCORerankDataset(Dataset):

    def __init__(
        self, 
        ret_query_data_path: str,
        ret_cand_data_path: str,
        annotation_path: str,
        image_path_prefix: str,
        split='val',
        type: str="query",
        rank_num: int=10
    ) -> None:

        super(GenecisCOCORerankDataset, self).__init__()
        self.type = type 
        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for item in self.val_samples:
            self.gallery_ids.add(str(item['target']['val_image_id']))
            gallery = item['gallery']
            for x in gallery:
                self.gallery_ids.add(str(x['val_image_id']))

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num

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
        instruction = instruction1
        instruction = "<instruction_start>" + instruction + "<instruction_end>"
        sample = self.val_samples[index // self.rank_num]
        reference_name = str(sample['reference']['val_image_id'])
        reference_img_path = f"{self.image_path_prefix}/{reference_name.zfill(12)}.jpg"
        relative_caption = sample['condition']
        relative_caption = format_string(relative_caption)
        # relative_caption = self.tokenizer(relative_caption, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # relative_caption = self.tokenizer.decode(relative_caption['input_ids'])
        relative_caption = f"{instruction} {relative_caption}"
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        target_name = str(sample['target']['val_image_id'])
        gallery_names = [str(item['val_image_id']) for item in sample['gallery']]
        target_and_gallery_names = [target_name]
        target_and_gallery_names.extend(gallery_names)
        cand_name = target_and_gallery_names[cand_id]
        cand_img_path = f"{self.image_path_prefix}/{cand_name.zfill(12)}.jpg"
        query_dict = {'image': reference_img_path, 'txt': relative_caption}
        cand_dict = {'image': cand_img_path}
        rerank_message = self.construct_rerank_messages(query_dict, cand_dict)
        return rerank_message

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 


class GenecisVAWRerankDataset(Dataset):

    def __init__(
        self, 
        ret_query_data_path: str,
        ret_cand_data_path: str, 
        annotation_path: str,
        image_path_prefix: str,
        type: str="query",
        rank_num: int=10
    ) -> None:

        super(GenecisVAWRerankDataset, self).__init__()
        self.type = type 

        self.val_samples = json.load(open(annotation_path))
        self.gallery_ids = set()
        for index, item in enumerate(self.val_samples):
            self.gallery_ids.add(f"{str(item['target']['image_id'])}_{index}_1.jpg")
            gallery = item['gallery']
            for i, x in enumerate(gallery):
                self.gallery_ids.add(f"{str(x['image_id'])}_{index}_{2 + i}.jpg")

        self.gallery_ids = sorted(list(self.gallery_ids))
        self.image_path_prefix = image_path_prefix

        self.ret_query_data = json.load(open(ret_query_data_path))
        self.ret_cand_data = json.load(open(ret_cand_data_path))
        self.rank_num = rank_num 

    def __len__(self) -> int:
        return len(self.ret_query_data) * self.rank_num 

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
        if 'change_attribute' in self.image_path_prefix:
            instruction = instruction1
            instruction = "<instruction_start>" + instruction + "<instruction_end>"
        elif 'focus_attribute' in self.image_path_prefix:
            instruction = instruction2
            instruction = "<instruction_start>" + instruction + "<instruction_end>"
        sample = self.val_samples[index // self.rank_num]
        reference_name = str(sample['reference']['image_id'])
        reference_img_path = f"{self.image_path_prefix}/{reference_name}_{index // self.rank_num}_0.jpg"
        relative_caption = sample['condition']
        relative_caption = format_string(relative_caption)
        # relative_caption = self.tokenizer(relative_caption, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # relative_caption = self.tokenizer.decode(relative_caption['input_ids'])
        relative_caption = f"{instruction} {relative_caption}"
        cand_idx = self.ret_query_data.index(index // self.rank_num)
        cand_id = self.ret_cand_data[cand_idx][index % self.rank_num]
        target_name = f"{str(sample['target']['image_id'])}_{index // self.rank_num}_1.jpg"
        gallery_names = [f"{str(item['image_id'])}_{index // self.rank_num}_{2 + idx}.jpg" for idx, item in enumerate(sample['gallery'])]
        target_and_gallery_names = [target_name]
        target_and_gallery_names.extend(gallery_names) 
        cand_name = target_and_gallery_names[cand_id]
        cand_img_path = f"{self.image_path_prefix}/{cand_name}"
        query_dict = {'image': reference_img_path, 'txt': relative_caption}
        cand_dict = {'image': cand_img_path}
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