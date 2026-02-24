import os
import re
import json
from torch.utils.data import Dataset
from typing import Dict, List # debug
import random 
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler,rank0_print_nested_dict,
)
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000
prompt1 = ["\nSummarize above image and sentence in one word: ","\nSummarize above sentence in one word: ","\nSummarize above image in one word: "]
prompt2 =["","",""]
all_prompt = [prompt1,prompt2]
IND = {
    "Classification": ["VOC2007", "N24News", "SUN397", "ImageNet_1K", "HatefulMemes"],
    "VQA": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"],
    "Retrieval": ["MSCOCO_i2t", "MSCOCO_t2i", "VisDial", "CIRR", "VisualNews_i2t", "VisualNews_t2i", "NIGHTS", "WebQA"],
    "Visual Grounding": ["MSCOCO"]
}
IND = [item for sublist in IND.values() for item in sublist]  # 扁平化 IND 列表
datasetid_IND = {dataset:index for index, dataset in enumerate(IND)}  # 创建 datasetid_IND 字典
IND_datasetid = {index:dataset for index, dataset in enumerate(IND)}  # 创建 index_datasetid 字典

class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        tokenizer = None,
        has_instruction=True,
        use_instruction_token = False, 
        has_hard_negative=False,
        has_modality_hard_negative=False,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        
        # 有时间可看一下这些数据加载完成都是什么格式 ！！！
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_json(instructions_path) # 这是一个字典形式
        self.tokenizer = tokenizer 
        self.image_path_prefix = image_path_prefix
        self.has_instruction = True # MBEIR 数据集中存在指令
        self.prompt = prompt2 # 咒语为空字符串
        self.use_instruction_token = use_instruction_token
        self.has_hard_negative = has_hard_negative
        self.has_modality_hard_negative = has_modality_hard_negative
        rank0_print("当前使用的咒语是: ", self.prompt)
        rank0_print("当前数据集是否存在指令: ", self.has_instruction)
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("Instruction token is enabled but the dataset does not have instructions.")
        rank0_print("query_data[0]",self.query_data[0],"len(query_data)",len(self.query_data))
        rank0_print("len(cand_pool)",len(self.cand_pool))
        rank0_print("cand_pool[0]: ",list(self.cand_pool.keys())[0],self.cand_pool[list(self.cand_pool.keys())[0]])

    def __len__(self) -> int:
        return len(self.query_data)

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
    
    # 获取实例，传进来一个索引，返回一个 instance 字典
    def get_instance(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)
        query_prompt = self.query_instructions.get(IND_datasetid[int(query_dataset_id)]).get("qry_instruction")
        # 添加指令特殊 token -----------------------------------------------------------------
        if self.use_instruction_token:
            query_prompt = "<instruction_start>" + query_prompt + "<instruction_end>"
        # -----------------------------------------------------------------------------------
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(f"{query_txt}")
        pos_img_path = pos_cand.get("img_path", None)

        # truncation processing is applied to prevent memory overflow.
        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])      
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])
        
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        instance = {"query": query}
        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None), # 否则返回 None
            self.image_path_prefix,
        )
        instance.update({"pos_cand": pos_cand})
        # 添加指令mask -----------------------------------------------------------------
        instance.update({"instruction": query_prompt})
        # -----------------------------------------------------------------------------
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        instruction_message = dict()
        instruction = instance['instruction']
        instruction_message.update({"instruction": instruction})
        query_message = self.construct_messages(query_dict)
        cand_message = self.construct_messages(cand_dict)
        return query_message, cand_message, instruction_message
# ---------------------------------------------------------------------------------------------
# QueryDataset 和 CandidateDataset 类的用于评估训练集结果
subset_names = [
    "N24News", 
    "MSCOCO_t2i", "A-OKVQA", "ChartQA", "CIRR", "Country211", "DocVQA", "EDIS",
    "FashionIQ", "GQA", "HatefulMemes", "ImageNet-1K", "ImageNet-A", "ImageNet-R",
    "InfographicsVQA", "MSCOCO", "MSCOCO_i2t",  "NIGHTS",
    "ObjectNet", "OK-VQA", "OVEN", "Place365", "RefCOCO", "RefCOCO-Matching", "ScienceQA", 
    "SUN397", "TextVQA", "VisDial", "Visual7W", "Visual7W-Pointing",
    "VisualNews_i2t", "VisualNews_t2i", "VizWiz", "VOC2007", "WebQA", "Wiki-SS-NQ",
]
test_task_categories = {
    'Classification': ['VOC2007', 'N24News', 'SUN397', 'ImageNet-1K', 'HatefulMemes', 'ObjectNet', 'Country211', 'Place365', 'ImageNet-A', 'ImageNet-R'],
    'VQA': ['OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'ScienceQA', 'GQA', 'TextVQA', 'VizWiz', 'Visual7W'],
    'Retrieval': ['MSCOCO_i2t', 'MSCOCO_t2i', 'VisDial', 'CIRR', 'VisualNews_i2t', 'VisualNews_t2i', 'NIGHTS', 'WebQA', 'Wiki-SS-NQ', 'FashionIQ', 'OVEN', 'EDIS'],
    'Visual_Grounding': ['MSCOCO', 'RefCOCO', 'RefCOCO-Matching', 'Visual7W-Pointing']
}
Classification = test_task_categories['Classification']
VQA = test_task_categories['VQA']
Retrieval = test_task_categories['Retrieval']
Visual_Grounding = test_task_categories['Visual_Grounding']
IND_categories = {
    "Classification": ["VOC2007", "N24News", "SUN397", "ImageNet-1K", "HatefulMemes"],
    "VQA": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"],
    "Retrieval": ["MSCOCO_i2t", "MSCOCO_t2i", "VisDial", "CIRR", "VisualNews_i2t", "VisualNews_t2i", "NIGHTS", "WebQA"],
    "Visual_Grounding": ["MSCOCO"]
}
OOD_categories = {
    "Classification": ["ObjectNet", "Country211", "Place365", "ImageNet-A", "ImageNet-R"],
    "VQA": ["ScienceQA", "GQA", "TextVQA", "VizWiz"],
    "Retrieval": ["Wiki-SS-NQ", "FashionIQ", "OVEN", "EDIS"],
    "Visual_Grounding": ["RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]
}
IND = [item for sublist in IND_categories.values() for item in sublist]
OOD = [item for sublist in OOD_categories.values() for item in sublist]
Overall = IND + OOD
assert set(Overall) == set(Classification + VQA + Retrieval + Visual_Grounding)
assert set(Overall) == set(subset_names)
datasetid_Overall = {dataset: index for index, dataset in enumerate(Overall)}
Overall_datasetid = {index: dataset for index, dataset in enumerate(Overall)}

class QueryDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        use_instruction_token = False, # 是否使用指令 token
        has_instruction = True, # 数据集是否存在指令
        prompt_index =1,

    ) -> None:
        super(QueryDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_json(instructions_path) # 这是一个字典形式
        self.image_path_prefix = image_path_prefix
        assert isinstance(prompt_index, int), "prompt_index must be an integer"
        self.prompt = all_prompt[prompt_index] # 默认咒语为空字符串
        self.use_instruction_token = use_instruction_token
        self.has_instruction = has_instruction # MBEIR 数据集中存在指令
        rank0_print("当前使用的咒语是:", self.prompt)
        rank0_print("当前数据集是否存在指令: ", self.has_instruction)
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("Instruction token is enabled but the dataset does not have instructions.") 

    def __len__(self) -> int:
        return len(self.query_data)

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

    def get_instance(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 
        query_prompt = self.query_instructions.get(Overall_datasetid[int(query_dataset_id)]).get("qry_instruction")
        # 添加指令特殊 token -----------------------------------------------------------------
        if self.use_instruction_token:
            query_prompt = "<instruction_start>" + query_prompt + "<instruction_end>"
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_without_prompt = format_string(f"{query_txt}")
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        instance = {"query": query}
        instance['query']['qid'] = hash_qid(qid)
        instance['instruction'] = query_prompt
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query = instance['query']
        qid = query['qid']
        query_message = self.construct_messages(query)
        instruction_message = dict()
        instruction = instance['instruction']
        instruction_message.update({"instruction": instruction})
        if self.has_instruction:
            return query_message, qid, instruction_message
        else:
            return query_message, qid

class CandidateDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str, 
        prompt_index = 1,
    ) -> None:
        super(CandidateDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool(cand_pool_path)
        self.query_instructions = _load_json(instructions_path) # 这是一个字典形式
        self.image_path_prefix = image_path_prefix
        assert isinstance(prompt_index, int), "prompt_index must be an integer"
        self.prompt = all_prompt[prompt_index] # 默认咒语为空字符串
        rank0_print("当前使用的咒语是:", self.prompt) 

    def __len__(self) -> int:
        return len(self.cand_pool)

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

    def get_instance(self, index):
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)
        img = _load_and_preprocess_image(img_path, self.image_path_prefix)
        did = mbeir_cand_pool_entry.get("did", None)
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        if cand_txt == "":
            cand_txt = None
        if img == "":
            img = None
        
        if img is not None and cand_txt is not None:
            instance = {
                "txt": cand_txt,
                "image": img, 
            }
        elif img is not None:
            instance = {
                "image": img, 
            }
        else:
            instance = {
                "txt": cand_txt,
            }
        instance.update({"did": hash_did(did)})
        return instance 
    
    def __getitem__(self, i):
        candidate = self.get_instance(i)
        did = candidate['did']
        candidate_message = self.construct_messages(candidate)
        return candidate_message, did

# 用于评估测试集结果 ----------------------------------------------------------------------------------------------
subset_names = [
    "N24News", "MSCOCO_t2i","A-OKVQA", "ChartQA","CIRR", "Country211", "DocVQA", "EDIS",
    "FashionIQ", "GQA", "HatefulMemes", "ImageNet-1K", "ImageNet-A", "ImageNet-R",
    "InfographicsVQA","MSCOCO", "MSCOCO_i2t",  "NIGHTS",
    "ObjectNet", "OK-VQA", "OVEN", "Place365", "RefCOCO", "RefCOCO-Matching", "ScienceQA", 
    "SUN397", "TextVQA", "VisDial", "Visual7W", "Visual7W-Pointing",
    "VisualNews_i2t", "VisualNews_t2i", "VizWiz", "VOC2007", "WebQA", "Wiki-SS-NQ"
]
test_task_categories = {'Classification': ['VOC2007', 'N24News', 'SUN397', 'ImageNet-1K', 'HatefulMemes', 'ObjectNet', 'Country211', 'Place365', 'ImageNet-A', 'ImageNet-R'], 
'VQA': ['OK-VQA', 'A-OKVQA', 'DocVQA', 'InfoVQA', 'ChartQA', 'ScienceQA', 'GQA', 'TextVQA', 'VizWiz'], 
'Retrieval': ['MSCOCO-i2t', 'MSCOCO-t2i', 'VisDial', 'CIRR', 'VisualNews-i2t', 'VisualNews-t2i', 'NIGHTS', 'WebQA', 'Wiki-SS-NQ', 'FashionIQ', 'OVEN', 'EDIS'], 
'Visual_Grounding': ['MSCOCO', 'RefCOCO', 'RefCOCO-Matching', 'Visual7W-Pointing']}
Classification = test_task_categories['Classification']
VQA = test_task_categories['VQA']
Retrieval = test_task_categories['Retrieval']
Visual_Grounding = test_task_categories['Visual_Grounding']

class MMEBEvalDataset(Dataset):
    """
    评估阶段的数据集，支持文本-图像检索任务
    功能：加载评估数据、去重文本-图像对、预处理图像和文本
    """
    def __init__(self,
                subset: str,
                text_field: str,
                img_path_field: str, 
                tokenizer,
                has_instruction=True,
                use_instruction_token=True, 
                ):
        """
        参数:
            subset (str): 36 个数据集的名称
            text_field (str): 数据中表示文本的字段名（如'qry_text','tgt_text'）
            img_path_field (str): 数据中表示图像路径的字段名（如'qry_img_path','tgt_img_path'）
        """
        super(MMEBEvalDataset, self).__init__()
        self.subset = subset
        self.text_field = text_field
        self.img_path_field = img_path_field
        self.tokenizer = tokenizer
        self.image_path_prefix = "/group/40077/Retrieval_Dataset/MMEB-eval/eval_images"
        self.prompt = all_prompt[1]
        self.use_instruction_token = use_instruction_token
        self.has_instruction = has_instruction
        rank0_print(f"{self.text_field} 当前使用的咒语是:", self.prompt)
        rank0_print(f"{self.text_field} 当前数据集是否存在指令: ", self.has_instruction)
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("Instruction token is enabled but the dataset does not have instructions.")

        assert subset in subset_names, f"Invalid subset name: {subset}. Must be one of {subset_names}."
        assert text_field in ["qry_text", "tgt_text"], f"Invalid text field: {text_field}. Must be 'qry_text' or 'tgt_text'."
        assert img_path_field in ["qry_img_path", "tgt_img_path"], f"Invalid image path field: {img_path_field}. Must be 'qry_img_path' or 'tgt_img_path'."
        # 加载评估数据集 /group/40077/Retrieval_Dataset/MMEB-eval/N24News/N24News_test.jsonl
        dataset_name = "/group/40077/Retrieval_Dataset/MMEB-eval"
        self.eval_data = _load_data_jsonl(os.path.join(dataset_name,subset + "/" + subset + "_test.jsonl"))
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        rank0_print(f"{self.text_field} Loaded {len(self.paired_data)} entries from {subset} dataset.")
        # "/group/40077/Retrieval_Dataset/MMEB-eval/data_instruction.json"
        instructions = _load_json(os.path.join(dataset_name, "data_instruction.json"))
        self.instruction = instructions[subset] if subset in instructions else None
        # 打印 paired_data 的第一条数据
        # rank0_print(f"{self.text_field} eval_data 的第一条数据:", self.eval_data[0])
        # rank0_print(f"{self.text_field} paired_data 的第一条数据:", self.paired_data[0])
        # 打印指令
        rank0_print(f"{self.text_field} 指令内容:", self.instruction)
        self.ids_all = set()

    def get_paired_data(self, text_field, img_path_field):
        """
        处理原始数据，生成唯一的文本-图像对，支持多种数据格式
        参数:
            text_field (str): 文本字段名
            img_path_field (str): 图像路径字段名
        返回:
            list: 去重后的文本-图像对列表（字典形式）
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:  # 非空文本
                    if isinstance(row[img_path_field], list):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], list):
                        for img_path in row[img_path_field]:
                            unique_pair.add(("", img_path))
                    else:
                        unique_pair.add(("", row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field]), \
                    "文本和图像列表长度必须一致"
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))
        # paired_data = [{"text": text, "image": img_path} for text, img_path in unique_pair]
        # 添加 id 索引
        paired_data = [{"text": text, "image": img_path, "id":idx} for idx,(text, img_path) in enumerate(unique_pair)]
        return paired_data
    def __len__(self):
        return len(self.paired_data)  # 返回去重后的样本总数
    
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

    def get_instance(self, index):
        entry = self.paired_data[index]
        # entry 的数据样式为 {"text": "文本内容", "image": "图像路径"}, 如果不存在则为 "" .
        txt = entry.get('text') or ""
        if "<|image_1|>" in txt:
            txt = txt.replace("<|image_1|>", "")
        img_path = entry.get('image') or ""
        if img_path == "":
            img_path = None
        if self.text_field == "qry_text":
            qry_instruction = self.instruction.get("qry_instruction")
            qry_txt = format_string(validate_string_start(txt.replace(qry_instruction, "").strip()))
            # 添加指令特殊 token -----------------------------------------------------------------
            if self.use_instruction_token:
                qry_instruction = "<instruction_start>" + qry_instruction + "<instruction_end>"
            # -----------------------------------------------------------------------------------
            qry_txt_with_instruction = format_string(f"{qry_instruction} {qry_txt}")
            qry_txt_without_instruction = format_string(f"{qry_txt}")
            if self.has_instruction:
                qry_txt_with_instruction = self.tokenizer(qry_txt_with_instruction, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                qry_txt_with_instruction = self.tokenizer.decode(qry_txt_with_instruction['input_ids'])
                query = _prepare_data_dict(qry_txt_with_instruction, img_path, self.image_path_prefix)
            else:
                qry_txt_without_instruction = self.tokenizer(qry_txt_without_instruction, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                qry_txt_without_instruction = self.tokenizer.decode(qry_txt_without_instruction['input_ids'])
                query = _prepare_data_dict(qry_txt_without_instruction, img_path, self.image_path_prefix)
            instance = {"query": query}
            instance['instruction'] = qry_instruction
        elif self.text_field == "tgt_text":
            tgt_instruction = self.instruction.get("tgt_instruction")
            tgt_txt = format_string(validate_string_start(txt.replace(tgt_instruction, "").strip()))
            # 添加指令特殊 token -----------------------------------------------------------------
            if self.use_instruction_token:
                tgt_instruction = "<instruction_start>" + tgt_instruction + "<instruction_end>"
            # -----------------------------------------------------------------------------------
            tgt_txt_with_instruction = format_string(f"{tgt_instruction} {tgt_txt}")
            tgt_txt_without_instruction = format_string(f"{tgt_txt}")
            if self.has_instruction:
                tgt_txt_with_instruction = self.tokenizer(tgt_txt_with_instruction, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                tgt_txt_with_instruction = self.tokenizer.decode(tgt_txt_with_instruction['input_ids'])
                query = _prepare_data_dict(tgt_txt_with_instruction, img_path, self.image_path_prefix)
            else:
                tgt_txt_without_instruction = self.tokenizer(tgt_txt_without_instruction, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                tgt_txt_without_instruction = self.tokenizer.decode(tgt_txt_without_instruction['input_ids'])
                query = _prepare_data_dict(tgt_txt_without_instruction, img_path, self.image_path_prefix)
            instance = {"query": query}
            instance['instruction'] = tgt_instruction
        else:
            raise ValueError(f"Invalid text field: {self.text_field}. Must be 'qry_text' or 'tgt_text'.")
        # if index == 0:
            # rank0_print(f"{self.text_field} 第一条数据 query:", query)
            # rank0_print(f"{self.text_field} instruction:", instance['instruction'])
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query = instance['query']
        query_message = self.construct_messages(query)
        instruction_message = dict()
        instruction = instance['instruction']
        instruction_message.update({"instruction": instruction})
        ids = self.paired_data[i]["id"]
        # if  i == 0:
        #     rank0_print(f"{self.text_field} 第一条数据 query_message:", query_message)
        #     rank0_print(f"{self.text_field} instruction_message:", instruction_message)
        return query_message, ids,instruction_message


def _load_data(data_path):
    """Validate and load data."""
    assert os.path.exists(data_path), f"Data Path {data_path} does not exist"
    assert data_path.endswith(".jsonl"), f"Data Path {data_path} is not a jsonl file"
    data_entries = _load_data_jsonl(data_path)
    return data_entries

def _load_query_data(query_data_path):
    query_data = _load_data(query_data_path)
    return query_data

def _load_cand_pool_as_dict(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    cand_pool_dict = {}
    for cand_pool_entry in cand_pool:
        did = cand_pool_entry.get("did")
        assert did, f"Cannot find did for {cand_pool_entry}"
        cand_pool_dict[did] = cand_pool_entry
    cand_pool = cand_pool_dict
    return cand_pool 

def _load_query_instructions(instructions_path):
    """Validate and load instructions."""
    # Validate the path and file extension
    assert os.path.exists(instructions_path), f"Instructions Path {instructions_path} does not exist"
    assert instructions_path.endswith(".tsv"), f"Instructions Path {instructions_path} is not a tsv file"
    prompts_dict = {}
    with open(instructions_path, "r") as f:
        next(f)  # Skip the header line
        for line in f.readlines():
            parts = line.strip().split("\t")
            # Construct the key to be dataset_id, query_modality, cand_modality
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
            prompts_dict[key] = prompts
    query_instructions = prompts_dict
    return query_instructions 

def _get_random_cand(cand_list):
    return random.choice(cand_list)

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s

def _get_random_query_prompt(dataset_id, query_modality, cand_modality, query_instructions):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = query_instructions.get(key, [])
    assert prompts, f"Cannot find prompts for {key}"
    prompt = format_string(random.choice(prompts))
    assert prompt, f"Prompt is empty for {key}"
    return prompt

def _load_and_preprocess_image(query_img_path, image_path_prefix):
    """Load an image given a path"""
    if not query_img_path:
        return None
    full_query_img_path = os.path.join(image_path_prefix, query_img_path)
    assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
    return full_query_img_path

def _prepare_data_dict(txt, img_path, image_path_prefix):
    img = _load_and_preprocess_image(img_path, image_path_prefix)
    if img is None or img == '':
        return {'txt': txt}
    elif txt == '':
        return {'image': img}
    return {"txt": txt, "image": img}

def _load_data_jsonl(datapath):
    data_entries = []
    with open(datapath, "r") as fin:
        for line in fin:
            data_entry = json.loads(line)
            data_entries.append(data_entry)
    return data_entries

def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def _load_cand_pool(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    return cand_pool
import re

def validate_string_start(input_str):
    """去除字符串开头的非法字符（空格、逗号、分号、句号、问号、冒号、换行符）"""
    return re.sub(r'^[\s,;:.?\n]+', '', input_str)

def _load_json(file_path):
    """
    加载 json 文件
    :param file_path: json 文件路径
    :return: json 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data