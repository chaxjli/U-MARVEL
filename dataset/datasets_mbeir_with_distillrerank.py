import os
import json
from torch.utils.data import Dataset
import copy
import torch
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
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
random.seed(42)
# fusion_results_random50.jsonl
# # 定义混合抽样策略
# topk_hard_neg_task_name = ["visualnews_task0","webqa_task1","edis_task2","webqa_task2","visualnews_task3","oven_task6",
#                           "cirr_task7","oven_task8"]
# lastk_hard_neg_task_name = ["mscoco_task0","fashion200k_task0","fashion200k_task3","fashioniq_task7"]
# randomk_neg_task_name = ["mscoco_task3","nights_task4","infoseek_task6","infoseek_task8"]

topk_hard_neg_task_name = ["visualnews_task0","webqa_task1","edis_task2","webqa_task2","visualnews_task3","oven_task6",
                           "cirr_task7","oven_task8","mscoco_task0","fashion200k_task0","fashion200k_task3","fashioniq_task7",
                           "mscoco_task3","nights_task4","infoseek_task6","infoseek_task8",
                           ]
lastk_hard_neg_task_name = []
randomk_neg_task_name = []
assert len(set(topk_hard_neg_task_name)) + len(set(lastk_hard_neg_task_name)) + len(set(randomk_neg_task_name)) ==  16, "数据集名称或者数量不匹配"
datasetid2name = {
    0: 'visualnews', 1: 'fashion200k', 2: 'webqa', 3: 'edis', 
    4: 'nights', 5: 'oven', 6: 'infoseek', 7: 'fashioniq', 8: 'cirr', 9: 'mscoco'}

class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """
    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        hard_negative_path: str,
        modality_hard_negative_path: str,
        pos_sample_path: str,
        image_path_prefix: str,
        query_feature_path: str,         # query feature 地址
        cand_feature_path: str,          # cand feature 地址
        tokenizer = None,
        has_instruction=True,
        use_instruction_token = True, 
        has_hard_negative=False,
        has_modality_hard_negative=False,
        topk_hard_negative = 50,
        topk_modality_hard_negative = 50,
        has_feature_constraint=False,  # 是否使用特征约束, con-train 使用,用来约束当前模型 feature 靠近 epoch1 的 feature
        has_rerank_scores=True,        # 是否使用 rerank scores
        has_distill_with_pos=False,    # 是否使用正样本进行辅助蒸馏
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)      # [:1]加载查询数据集, debug 将数据集大小限制为 1
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path) # 加载训练集的候选池，这是个字典 key 是 did，value 是数据
        self.query_instructions = _load_query_instructions(instructions_path)
        self.tokenizer = tokenizer 
        self.image_path_prefix = image_path_prefix
        self.has_instruction = True # MBEIR 数据集中存在指令
        self.prompt = prompt2       # 咒语为空字符串
        self.use_instruction_token = use_instruction_token

        self.has_distill_with_pos = has_distill_with_pos  # 是否使用正样本进行辅助蒸馏
        if self.has_distill_with_pos:
            assert self.has_rerank_scores, "使用正样本进行辅助蒸馏时，必须使用 rerank scores"
        # 处理 dataset 的 local 和 global 的蒸馏数据 ------------------------------------------------------------
        self.has_hard_negative = has_hard_negative
        self.has_modality_hard_negative = has_modality_hard_negative
        self.topk_hard_negative = topk_hard_negative
        self.topk_modality_hard_negative = topk_modality_hard_negative
        self.has_feature_constraint = has_feature_constraint  # 是否使用特征约束
        # 处理 dataset 的 rerank scores 参数和数据 -------------------------------------------------------------
        self.has_rerank_scores = has_rerank_scores            # 是否蒸馏操作
        rank0_print("当前使用的咒语是: ", self.prompt)
        rank0_print("当前数据集是否存在指令: ", self.has_instruction)
        if self.use_instruction_token and not self.has_instruction:
            raise ValueError("Instruction token is enabled but the dataset does not have instructions.")

        # 加载正样本数据
        if self.has_distill_with_pos:
            assert os.path.exists(pos_sample_path), f"Pos Sample Path {pos_sample_path} does not exist"
            self.pos_sample_data = _load_data(pos_sample_path)
            rank0_print("len(pos_sample_data)",len(self.pos_sample_data))
        
        # 加载 local 的蒸馏数据
        if self.has_hard_negative:
            assert os.path.exists(hard_negative_path), f"Hard Negative Path {hard_negative_path} does not exist"
            self.hard_negative_data = _load_data(hard_negative_path)
            rank0_print("len(hard_negative_data)",len(self.hard_negative_data))
            # # 处理 local 随机抽样的 50 个 蒸馏数据集
            # random50_hard_negative_path = hard_negative_path.replace("fusion_results.jsonl", "fusion_results_random50.jsonl")
            # assert os.path.exists(random50_hard_negative_path), f"Random50 Hard Negative Path {random50_hard_negative_path} does not exist"
            # self.hard_negative_data_random50 = _load_data(random50_hard_negative_path)
        
        # 加载 global 的蒸馏数据
        if self.has_modality_hard_negative:
            assert os.path.exists(modality_hard_negative_path), f"Modality Hard Negative Path {modality_hard_negative_path} does not exist"
            self.modality_hard_negative_data = _load_data(modality_hard_negative_path)
            rank0_print("len(modality_hard_negative_data)",len(self.modality_hard_negative_data))
            # # 处理 global 随机抽样的 50 个 蒸馏数据集
            # random50_modality_hard_negative_path = modality_hard_negative_path.replace("fusion_results.jsonl", "fusion_results_random50.jsonl")
            # assert os.path.exists(random50_modality_hard_negative_path), f"Random50 Modality Hard Negative Path {random50_modality_hard_negative_path} does not exist"
            # self.modality_hard_negative_data_random50 = _load_data(random50_modality_hard_negative_path)
        
        if self.has_feature_constraint:
            assert os.path.exists(query_feature_path), f"Query Feature Path {query_feature_path} does not exist"
            assert os.path.exists(cand_feature_path), f"Candidate Feature Path {cand_feature_path} does not exist"
            self.query_feature = torch.load(query_feature_path)
            self.cand_feature = torch.load(cand_feature_path)
            rank0_print("len(query_feature)",len(self.query_feature))
            rank0_print("len(cand_feature)",len(self.cand_feature))

        rank0_print("query_data[0]",self.query_data[0],"len(query_data)",len(self.query_data))
        rank0_print("len(cand_pool)",len(self.cand_pool))
        rank0_print("cand_pool[0]: ",list(self.cand_pool.keys())[0],self.cand_pool[list(self.cand_pool.keys())[0]])

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
    
    # 获取实例，传进来一个索引，返回一个 instance 字典
    def get_instance(self, index):
        
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        # 处理正样本数据, 用于辅助蒸馏 todo
        if self.has_distill_with_pos:
            pos_sample_score = self.pos_sample_data[qid][selected_pos_cand_did]
            pos_index_list = [1]

        # 处理负样本数据
        datasetid = int(mbeir_entry['qid'].split(":")[0])
        task_id = int(mbeir_entry["task_id"])
        task_name = datasetid2name[datasetid]+"_task"+str(task_id)
        # 处理 local 负样本数据 ---------------------------------------------------------------------------------------------------------------
        if self.has_hard_negative:
            assert qid == self.hard_negative_data[index]["qid"], f"数据不匹配: {qid} != {self.hard_negative_data[index]['qid']}"
            local_dids = self.hard_negative_data[index]["did"]
            local_scores = self.hard_negative_data[index]["score"]
            assert len(local_dids) == len(local_scores), f"数据不匹配: {len(local_dids)} != {len(local_scores)}"
            if task_name in topk_hard_neg_task_name:
                topk_hard_negative_did_list = local_dids[:self.topk_hard_negative]
                topk_hard_negative_score_list = local_scores[:self.topk_hard_negative]

                # 处理正样本索引矩阵
                if self.has_distill_with_pos:
                    for did in topk_hard_negative_did_list:
                        if did in pos_cand_list:
                            pos_index_list.append(1)
                        else:
                            pos_index_list.append(0)
            
            elif task_name in lastk_hard_neg_task_name:
                topk_hard_negative_did_list = local_dids[-self.topk_hard_negative:]
                topk_hard_negative_score_list = local_scores[-self.topk_hard_negative:]
            elif task_name in randomk_neg_task_name:
                # self.hard_negative_data_random50 从这个里面获取数据
                assert qid == self.hard_negative_data_random50[index]["qid"], f"数据不匹配: {qid} != {self.hard_negative_data_random50[index]['qid']}"
                local_dids_random = self.hard_negative_data_random50[index]["did"]
                local_scores_random = self.hard_negative_data_random50[index]["score"]
                assert len(local_dids_random) == len(local_scores_random), f"数据不匹配: {len(local_dids_random)} != {len(local_scores_random)}"
                topk_hard_negative_did_list = local_dids_random[:self.topk_hard_negative]
                topk_hard_negative_score_list = local_scores_random[:self.topk_hard_negative]

            else:
                raise ValueError(f"Invalid task name: {task_name}")

        
        # 处理 global 负样本数据 ---------------------------------------------------------------------------------------------------------------
        if self.has_modality_hard_negative:
            assert qid == self.modality_hard_negative_data[index]["qid"], f"数据不匹配: {qid} != {self.modality_hard_negative_data[index]['qid']}"
            global_dids = self.modality_hard_negative_data[index]["did"]
            global_scores = self.modality_hard_negative_data[index]["score"]
            assert len(global_dids) == len(global_scores), f"数据不匹配: {len(global_dids)} != {len(global_scores)}"
            if task_name in topk_hard_neg_task_name:
                topk_modality_hard_negative_did_list = global_dids[:self.topk_modality_hard_negative]
                topk_modality_hard_negative_score_list = global_scores[:self.topk_modality_hard_negative]
                
                # 处理正样本索引矩阵
                if self.has_distill_with_pos:
                    for did in topk_modality_hard_negative_did_list:
                        if did in pos_cand_list:
                            pos_index_list.append(1)
                        else:
                            pos_index_list.append(0)
            
            elif task_name in lastk_hard_neg_task_name:
                topk_modality_hard_negative_did_list = global_dids[-self.topk_modality_hard_negative:]
                topk_modality_hard_negative_score_list = global_scores[-self.topk_modality_hard_negative:]
            elif task_name in randomk_neg_task_name:
                # self.modality_hard_negative_data_random50 从这个里面获取数据
                assert qid == self.modality_hard_negative_data_random50[index]["qid"], f"数据不匹配: {qid} != {self.modality_hard_negative_data_random50[index]['qid']}"
                global_dids_random = self.modality_hard_negative_data_random50[index]["did"]
                global_scores_random = self.modality_hard_negative_data_random50[index]["score"]
                assert len(global_dids_random) == len(global_scores_random), f"数据不匹配: {len(global_dids_random)} != {len(global_scores_random)}"
                topk_modality_hard_negative_did_list = global_dids_random[:self.topk_modality_hard_negative]
                topk_modality_hard_negative_score_list = global_scores_random[:self.topk_modality_hard_negative]
            else:
                raise ValueError(f"Invalid task name: {task_name}")

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
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
        # 添加 hard negative 数据集
        if self.has_hard_negative:
            topk_hard_negative_list = []
            for hard_negative_did in topk_hard_negative_did_list:
                neg_cand = self.cand_pool.get(hard_negative_did)
                neg_img_path = neg_cand.get("img_path", None)
                neg_cand_txt = neg_cand.get("txt") or ""
                neg_cand_txt = format_string(neg_cand_txt)
                neg_cand_txt = self.tokenizer(neg_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                neg_cand_txt = self.tokenizer.decode(neg_cand_txt['input_ids'])
                neg_cand = _prepare_data_dict(
                    neg_cand_txt,
                    neg_img_path,
                    self.image_path_prefix,
                )
                topk_hard_negative_list.append(neg_cand)
            instance.update({"topk_hard_negative": copy.deepcopy(topk_hard_negative_list)})
        # 添加 modality hard negative 数据集
        if self.has_modality_hard_negative:
            topk_modality_hard_negative_list = []
            for modality_hard_negative_did in topk_modality_hard_negative_did_list:
                neg_cand = self.cand_pool.get(modality_hard_negative_did)
                neg_img_path = neg_cand.get("img_path", None)
                neg_cand_txt = neg_cand.get("txt") or ""
                neg_cand_txt = format_string(neg_cand_txt)
                neg_cand_txt = self.tokenizer(neg_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
                neg_cand_txt = self.tokenizer.decode(neg_cand_txt['input_ids'])
                neg_cand = _prepare_data_dict(
                    neg_cand_txt,
                    neg_img_path,
                    self.image_path_prefix,
                )
                topk_modality_hard_negative_list.append(neg_cand)
            instance.update({"topk_modality_hard_negative": copy.deepcopy(topk_modality_hard_negative_list)})
        
        # 添加特征约束
        if self.has_feature_constraint:
            feature_list = []
            query_feature = self.query_feature[qid]
            cand_feature = self.cand_feature[selected_pos_cand_did]
            feature_list.append(query_feature)
            feature_list.append(cand_feature)
            if self.has_hard_negative:
                for hard_negative_did in topk_hard_negative_did_list:
                    hard_negative_feature = self.cand_feature[hard_negative_did]
                    feature_list.append(hard_negative_feature)
            if self.has_modality_hard_negative:
                for modality_hard_negative_did in topk_modality_hard_negative_did_list:
                    modality_hard_negative_feature = self.cand_feature[modality_hard_negative_did]
                    feature_list.append(modality_hard_negative_feature)
            instance.update({"feature": copy.deepcopy(feature_list)})
        
        # 添加 rerank scores
        if self.has_rerank_scores:
            rerank_score_list = []
            if self.has_distill_with_pos:
                rerank_score_list.append(pos_sample_score)
            if self.has_hard_negative:
                rerank_score_list.extend(topk_hard_negative_score_list)
            if self.has_modality_hard_negative:
                rerank_score_list.extend(topk_modality_hard_negative_score_list)
            instance.update({"scores": copy.deepcopy(rerank_score_list)})
        if self.has_distill_with_pos:
            instance.update({"pos_index": copy.deepcopy(pos_index_list)})
        return instance 

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        
        # 获取指令信息, 处理成字典格式
        instruction_message = dict()
        instruction = instance['instruction']
        instruction_message.update({"instruction": instruction})
        # 处理特征
        if self.has_feature_constraint:
            feature_list = instance['feature']
            feature_message = dict()
            feature_message.update({"feature": feature_list})
        
        # 处理分数
        if self.has_rerank_scores:
            rerank_score_list = instance['scores']
            rerank_score_message = dict()
            rerank_score_message.update({"scores": rerank_score_list})
        if self.has_distill_with_pos:
            pos_index_list = instance['pos_index']
            pos_index_message = dict()
            pos_index_message.update({"pos_index": pos_index_list})


        query_message = self.construct_messages(query_dict)
        cand_message = self.construct_messages(cand_dict)
        # 处理 hard negative 数据集
        if self.has_hard_negative:
            topk_hard_negative_message_list = []
            for hard_negative in instance["topk_hard_negative"]:
                hard_negative_message = self.construct_messages(hard_negative)
                topk_hard_negative_message_list.append(hard_negative_message)
        if self.has_modality_hard_negative:
            topk_modality_hard_negative_message_list = []
            for modality_hard_negative in instance["topk_modality_hard_negative"]:
                modality_hard_negative_message = self.construct_messages(modality_hard_negative)
                topk_modality_hard_negative_message_list.append(modality_hard_negative_message)
            
        result_list = []
        result_list.append(query_message)
        result_list.append(cand_message)
        if self.has_hard_negative:
            result_list.extend(topk_hard_negative_message_list)
        if self.has_modality_hard_negative:
            result_list.extend(topk_modality_hard_negative_message_list)
        
        if self.has_feature_constraint:
            result_list.append(feature_message)
        if self.has_rerank_scores:
            result_list.append(rerank_score_message)
        if self.has_instruction:
            result_list.append(instruction_message)
        if self.has_distill_with_pos:
            result_list.append(pos_index_message)
        result_tuple = tuple(result_list)
        return result_tuple
    

# QueryDataset 和 CandidateDataset 类的用于测试
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
        self.query_instructions = _load_query_instructions(instructions_path)
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
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 

        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list) # 随机选择一个正样本
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        # 添加指令特殊 token -----------------------------------------------------------------
        if self.use_instruction_token:
            query_prompt = "<instruction_start>" + query_prompt + "<instruction_end>"
        # -----------------------------------------------------------------------------------
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
    
    import copy
    def select(self, indices):
        """安全创建子数据集 (绕过文件路径校验)"""
        
        # 创建未初始化实例
        new_dataset = self.__class__.__new__(self.__class__)
        
        # 复制必要属性 (绕过__init__)
        new_dataset.image_path_prefix = self.image_path_prefix
        new_dataset.use_instruction_token = self.use_instruction_token
        new_dataset.has_instruction = self.has_instruction
        new_dataset.prompt = copy.deepcopy(self.prompt)
        # 直接注入已加载数据
        new_dataset.query_data = [copy.deepcopy(self.query_data[i]) for i in indices]
        new_dataset.cand_pool = self.cand_pool  # 共享候选池引用
        new_dataset.query_instructions = copy.deepcopy(self.query_instructions)
        if hasattr(self, '_is_initialized'):
            new_dataset._is_initialized = True
        # 验证关键属性
        assert len(new_dataset.query_data) == len(indices), "查询数据索引错误"
        assert new_dataset.cand_pool is self.cand_pool, "候选池未正确共享"
        assert new_dataset.query_instructions == self.query_instructions, "指令集不一致"
        return new_dataset


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
        self.query_instructions = _load_query_instructions(instructions_path)
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
        cand_modality = mbeir_cand_pool_entry.get("modality", None)
        if img is not None and cand_txt is not None:
            instance = {
                "txt": cand_txt,
                "image": img, 
                "modality": cand_modality,
            }
        elif img is not None:
            instance = {
                "image": img, 
                "modality": cand_modality,
            }
        else:
            instance = {
                "txt": cand_txt,
                "modality": cand_modality,
            }
        instance.update({"did": hash_did(did)})
        return instance 
    

    def __getitem__(self, i):
        candidate = self.get_instance(i)
        did = candidate['did']
        candidate_message = self.construct_messages(candidate)
        
        return candidate_message, did 


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
    if img is None:
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

def _load_json(file_path):
    """
    加载 json 文件
    :param file_path: json 文件路径
    :return: json 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def _select_topk_hard_neg(gt_docs,top100_docs,score,topk)->List:
    """
    选择 hard negative
    :param gt_docs: 正样本列表
    :param top100_docs: 候选池
    :param score: 分数
    :param topk: 选择的个数
    :return: 选择的 hard negative did 列表
    """
    selected_hard_neg = set()
    first_hard_neg_did = None
    assert topk > 0, "选取负样本数量必须大于 0"
    assert len(top100_docs) == len(score), "top100_docs 和 score 的长度不一致"
    assert len(top100_docs) > topk, "top100_docs 的长度必须大于 topk"
    for i in range(len(top100_docs)):
        if top100_docs[i] not in gt_docs:
            selected_hard_neg.add((top100_docs[i]))
            if first_hard_neg_did is None:
                first_hard_neg_did = top100_docs[i]
        if len(selected_hard_neg) >= topk:
            break
    assert first_hard_neg_did not in gt_docs, "first_hard_neg_did 不能在 gt_docs 中"
    selected_hard_neg = list(selected_hard_neg)
    if len(selected_hard_neg) < topk:
        selected_hard_neg.extend([first_hard_neg_did] * (topk - len(selected_hard_neg)))
    return selected_hard_neg



def _load_jsonl(file_path):
    """
    加载 jsonl 文件
    :param file_path: jsonl 文件路径
    :return: jsonl 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data