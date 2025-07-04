import os
import json
import torch
from torch.utils.data import Dataset
import random 
# 初始化下面分布式环境
# def init_distributed():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         torch.distributed.init_process_group(
#             backend='nccl',
#             init_method='env://',
#             world_size=world_size,
#             rank=rank
#         )
#         torch.cuda.set_device(local_rank)
#     else:
#         rank = 0
#         world_size = 1
#         local_rank = 0
#     return rank, world_size, local_rank

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

class LazySupervisedDataset(Dataset):
    """
    主类 LazySupervisedDataset 继承自 PyTorch 的 Dataset
    采用惰性加载策略，减少内存占用，支持多模态数据（文本+图像）
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        rerank_data_path: str, 
        image_path_prefix: str, 
        tokenizer = None,
        rank = 0,
        world_size = 1
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        # self.query_data = self.query_data
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.rerank_data = json.load(open(rerank_data_path))
        self.image_path_prefix = image_path_prefix
        self.tokenizer = tokenizer
        self.rank = rank             # 适配多机多卡训练
        self.world_size = world_size # 适配多机多卡训练
        # 数据分片，对数据进行切片是不是意味着可以调大一点 batch_size ???? 待定
        # 该切片操作确保每个进程获取不同的数据子集（例如，rank=0 取 0,8,16...；rank=1 取 1,9,17...）
        # 注意点：需确保 self.query_data 是 全局完整数据集 的分片，而不是每个进程已加载部分数据
        self.query_data = self.query_data[self.rank::self.world_size]

    def __len__(self) -> int:
        return len(self.query_data)

    def construct_rerank_messages_single_candidate(self, query_dict, cand_dict, type='pos'):
        """
        构建单候选重排序消息结构---- single candidate
        :param query_dict: 查询数据字典（包含文本/图像）
        :param cand_dict:  候选数据字典
        :param type: 候选类型（pos/neg）
        :return: 结构化消息列表
        """
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate\
                        meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, responed with 'No'."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes" if type == 'pos' else "No"}
                ]
            }
        ]
        # 动态构建查询和候选内容
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

    def construct_rerank_messages_multi_candidates(self, query_dict, cand_lists, ans):
        """
        构建多候选重排序消息结构
        :param query_dict: 查询数据字典
        :param cand_lists: 候选数据列表
        :param ans: 正确答案位置
        :return: 结构化消息列表
        """
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query followed by multiple candidates in the format: (1) cand1 (2) cand2, etc. Each candidate is independent of the others. \
                        Evaluate each candidate against the query, and respond with the number corresponding to the candidate that best meets the requirements of the query."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Ans: ({ans})"}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidates:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        for i, cand_dict in enumerate(cand_lists):
            cand.append({'type': 'text', 'text': f'({i + 1}) '})
            if 'image' in cand_dict:
                cand.append({'type': 'image', 'image': cand_dict['image']})
            if 'txt' in cand_dict:
                cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def get_instance(self, index, num_candidates):
        """
        获取单个训练实例
        :param index: 数据索引
        :param num_candidates: 候选数量
        :return: 包含正负样本的实例字典
        """
        # 获取基础数据
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

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        # truncation processing is applied to prevent memory overflow.
        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=256, padding=False, return_tensors=None, add_special_tokens=False)
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])
        query_txt_without_prompt = format_string(f"{query_txt}")
        pos_img_path = pos_cand.get("img_path", None)
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        # query = _prepare_data_dict(query_txt_without_prompt, query_img_path, image_path_prefix)
        instance = {"query": query}
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=256, padding=False, return_tensors=None, add_special_tokens=False)
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])
        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
            self.image_path_prefix,
        )
        instance.update({"pos_cand": pos_cand})

        # get the rerank data
        gt_docs = self.rerank_data[qid]['gt_docs']
        topk_docs = self.rerank_data[qid]['top100_docs']

        filtered_docs = [item for item in topk_docs if item not in gt_docs]
        if len(filtered_docs) == 0:
            neg_doc_id = random.choice(topk_docs)
        else:
            neg_doc_id = random.choice(filtered_docs)
        if len(filtered_docs) < num_candidates:
            neg_doc_ids = random.sample(topk_docs, num_candidates)
        else:
            neg_doc_ids = random.sample(filtered_docs, num_candidates)
        
        neg_cand = self.get_neg_cand(neg_doc_id)
        neg_cands = []
        for neg_id in neg_doc_ids:
            neg_cands.append(self.get_neg_cand(neg_id))
        instance.update({"neg_cand": neg_cand})
        instance.update({"neg_cand_lists": neg_cands})
        return instance 


    def get_neg_cand(self, neg_doc_id):
        neg_cand = self.cand_pool.get(neg_doc_id)
        neg_cand_txt = neg_cand.get("txt") or ""
        neg_cand_txt = format_string(neg_cand_txt)
        neg_cand_txt = self.tokenizer(neg_cand_txt, truncation=True, max_length=256, padding=False, return_tensors=None, add_special_tokens=False)
        neg_cand_txt = self.tokenizer.decode(neg_cand_txt['input_ids'])
        neg_image_path_prefix = self.image_path_prefix
        neg_cand = _prepare_data_dict(
            neg_cand_txt,
            neg_cand.get("img_path", None),
            neg_image_path_prefix,
        )
        return neg_cand 

    def __getitem__(self, i):
        
        # 各进程的 random 模块未同步种子，导致数据增强不一致, 使用PyTorch的分布式随机数生成器
        # num_candidates = random.randint(1, 4) # the current version considers up to five candidates at most.
        num_candidates = torch.randint(1, 5, (1,)).item()
        instance = self.get_instance(i, num_candidates)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        neg_dict = instance['neg_cand']
        cand_dict_lists = instance['neg_cand_lists']
        rerank_pos_message = self.construct_rerank_messages_single_candidate(query_dict, cand_dict, type='pos')
        rerank_neg_message = self.construct_rerank_messages_single_candidate(query_dict, neg_dict, type='neg')
        # generate random answer position
        # ans = random.randint(1, num_candidates + 1)
        ans = torch.randint(1, num_candidates + 2, (1,)).item()
        cand_dict_lists.insert(ans - 1, cand_dict)
        rerank_multi_candidates_message = self.construct_rerank_messages_multi_candidates(query_dict, cand_dict_lists, ans)
        return rerank_pos_message, rerank_neg_message, rerank_multi_candidates_message



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
