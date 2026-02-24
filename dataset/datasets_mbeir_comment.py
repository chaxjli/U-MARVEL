import os
import json
from torch.utils.data import Dataset
import random 
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

# 定义查询数据数量的上限
DATASET_QUERY_NUM_UPPER_BOUND = 500000
# 定义候选数据数量的上限
DATASET_CAN_NUM_UPPER_BOUND = 10000000

# 定义提示信息列表，用于在不同场景下向用户提问
prompt1 = ["\nSummarize above image and sentence in one word: ",
           "\nSummarize above sentence in one word: ",
           "\nSummarize above image in one word: "]
# 定义另一个提示信息列表，这里为空字符串
prompt2 = ["", "", ""]

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
        tokenizer = None 
    ) -> None:
        """
        初始化 LazySupervisedDataset 类。

        参数:
        query_data_path (str): 查询数据的文件路径。
        cand_pool_path (str): 候选池数据的文件路径。
        instructions_path (str): 指令数据的文件路径。
        image_path_prefix (str): 图像文件路径的前缀。
        tokenizer: 分词器，默认为 None。
        """
        super(LazySupervisedDataset, self).__init__()
        # 加载查询数据
        self.query_data = _load_query_data(query_data_path)
        # 加载候选池数据并转换为字典形式
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        # 加载查询指令
        self.query_instructions = _load_query_instructions(instructions_path)
        self.tokenizer = tokenizer 
        self.image_path_prefix = image_path_prefix
        # 使用 prompt2 作为提示信息
        self.prompt = prompt2 
        # 打印当前使用的提示信息
        rank0_print("当前使用的咒语是:", self.prompt)

    def __len__(self) -> int:
        """
        返回查询数据的长度。

        返回:
        int: 查询数据的长度。
        """
        return len(self.query_data)

    def construct_messages(self, data_dict):
        """
        根据数据字典构建消息列表。

        参数:
        data_dict (dict): 包含文本和图像信息的数据字典。

        返回:
        list: 包含用户消息和助手消息的列表。
        """
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
        """
        根据索引获取一个实例。

        参数:
        index (int): 索引。

        返回:
        dict: 包含查询和正样本候选的实例字典。
        """
        mbeir_entry = self.query_data[index]
        # 获取查询文本
        query_txt = mbeir_entry.get('query_txt') or ""
        # 获取查询图像路径
        query_img_path = mbeir_entry.get('query_img_path', None)
        # 获取查询的模态信息
        query_modality = mbeir_entry.get("query_modality", None)
        # 获取查询的 ID
        qid = mbeir_entry.get("qid", None)
        # 获取查询所属数据集的 ID
        query_dataset_id = qid.split(":")[0] if qid else None 
        # 获取正样本候选列表
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        # 随机选择一个正样本候选的 ID
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        # 根据 ID 从候选池中获取正样本候选
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        # 获取正样本候选所属数据集的 ID
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        # 获取正样本候选的模态信息
        pos_cand_modality = pos_cand.get("modality", None)
        # 获取正样本候选的文本
        pos_cand_txt = pos_cand.get("txt") or ""
        # 格式化正样本候选的文本
        pos_cand_txt = format_string(pos_cand_txt)

        # 获取随机的查询提示
        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        # 将查询提示和查询文本组合并格式化
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        # 格式化查询文本
        query_txt_without_prompt = format_string(f"{query_txt}")
        # 获取正样本候选的图像路径
        pos_img_path = pos_cand.get("img_path", None)

        # 对查询提示进行分词处理，并进行截断操作
        instruction_token_id = self.tokenizer(query_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # 对包含提示的查询文本进行分词处理，并进行截断操作
        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # 将分词后的 ID 转换为文本
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])
        # 对正样本候选的文本进行分词处理，并进行截断操作
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        # 将分词后的 ID 转换为文本
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])
        
        # 准备查询数据字典
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        instance = {"query": query}
        # 准备正样本候选数据字典
        pos_cand = _prepare_data_dict(
            pos_cand_txt,
            pos_cand.get("img_path", None),
            self.image_path_prefix,
        )
        instance.update({"pos_cand": pos_cand})
        return instance 

    def __getitem__(self, i):
        """
        根据索引获取查询消息和正样本候选消息。

        参数:
        i (int): 索引。

        返回:
        tuple: 包含查询消息和正样本候选消息的元组。
        """
        instance = self.get_instance(i)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        # 构建查询消息
        query_message = self.construct_messages(query_dict)
        # 构建正样本候选消息
        cand_message = self.construct_messages(cand_dict)

        return query_message, cand_message

    # 自定义的 select 方法，用于截取数据集的一部分
    def select(self, indices):
        """
        截取数据集的一部分。

        参数:
        indices (list): 要截取的索引列表。

        返回:
        LazySupervisedDataset: 新的数据集实例。
        """
        # 创建一个新的查询数据列表，只包含指定索引的数据
        new_query_data = [self.query_data[i] for i in indices]
        # 创建一个新的 LazySupervisedDataset 实例
        new_dataset = LazySupervisedDataset(
            query_data_path="",  # 这里可以传入空字符串，因为新的查询数据已经提取出来
            cand_pool_path="",  # 这里可以传入空字符串，因为候选池数据不变
            instructions_path="",  # 这里可以传入空字符串，因为指令数据不变
            image_path_prefix=self.image_path_prefix,
            tokenizer=self.tokenizer
        )
        # 更新新数据集的查询数据
        new_dataset.query_data = new_query_data
        return new_dataset

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
    ) -> None:
        """
        初始化 QueryDataset 类。

        参数:
        query_data_path (str): 查询数据的文件路径。
        cand_pool_path (str): 候选池数据的文件路径。
        instructions_path (str): 指令数据的文件路径。
        image_path_prefix (str): 图像文件路径的前缀。
        """
        super(QueryDataset, self).__init__()
        # 加载查询数据
        self.query_data = _load_query_data(query_data_path)
        # 加载候选池数据并转换为字典形式
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        # 加载查询指令
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix = image_path_prefix
        # 使用 prompt2 作为提示信息
        self.prompt = prompt2 
        # 打印当前使用的提示信息
        rank0_print("当前使用的咒语是:", self.prompt) 

    def __len__(self) -> int:
        """
        返回查询数据的长度。

        返回:
        int: 查询数据的长度。
        """
        return len(self.query_data)

    def construct_messages(self, data_dict):
        """
        根据数据字典构建消息列表。

        参数:
        data_dict (dict): 包含文本和图像信息的数据字典。

        返回:
        list: 包含用户消息和助手消息的列表。
        """
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
        """
        根据索引获取一个实例。

        参数:
        index (int): 索引。

        返回:
        dict: 包含查询的实例字典。
        """
        mbeir_entry = self.query_data[index]
        # 获取查询文本
        query_txt = mbeir_entry.get('query_txt') or ""
        # 获取查询图像路径
        query_img_path = mbeir_entry.get('query_img_path', None)
        # 获取查询的模态信息
        query_modality = mbeir_entry.get("query_modality", None)
        # 获取查询的 ID
        qid = mbeir_entry.get("qid", None)
        # 获取查询所属数据集的 ID
        query_dataset_id = qid.split(":")[0] if qid else None 

        # 获取正样本候选列表
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        # 随机选择一个正样本候选的 ID
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        # 根据 ID 从候选池中获取正样本候选
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        # 获取正样本候选所属数据集的 ID
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        # 获取正样本候选的模态信息
        pos_cand_modality = pos_cand.get("modality", None)
        # 获取正样本候选的文本
        pos_cand_txt = pos_cand.get("txt") or ""
        # 格式化正样本候选的文本
        pos_cand_txt = format_string(pos_cand_txt)

        # 获取随机的查询提示
        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        # 将查询提示和查询文本组合并格式化
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        # 格式化查询文本
        query_txt_without_prompt = format_string(f"{query_txt}")

        # 准备查询数据字典
        query = _prepare_data_dict(query_txt_with_prompt, query_img_path, self.image_path_prefix)
        instance = {"query": query}
        # 对查询的 ID 进行哈希处理
        instance['query']['qid'] = hash_qid(qid)
        return instance 

    def __getitem__(self, i):
        """
        根据索引获取查询消息和查询的哈希 ID。

        参数:
        i (int): 索引。

        返回:
        tuple: 包含查询消息和查询的哈希 ID 的元组。
        """
        instance = self.get_instance(i)
        query = instance['query']
        qid = query['qid']
        # 构建查询消息
        query_message = self.construct_messages(query)
        
        return query_message, qid 

class CandidateDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str, 
    ) -> None:
        """
        初始化 CandidateDataset 类。

        参数:
        query_data_path (str): 查询数据的文件路径。
        cand_pool_path (str): 候选池数据的文件路径。
        instructions_path (str): 指令数据的文件路径。
        image_path_prefix (str): 图像文件路径的前缀。
        """
        super(CandidateDataset, self).__init__()
        # 加载查询数据
        self.query_data = _load_query_data(query_data_path)
        # 加载候选池数据
        self.cand_pool = _load_cand_pool(cand_pool_path)
        # 加载查询指令
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix = image_path_prefix
        # 使用 prompt2 作为提示信息
        self.prompt = prompt2 
        # 打印当前使用的提示信息
        rank0_print("当前使用的咒语是:", self.prompt) 

    def __len__(self) -> int:
        """
        返回候选池数据的长度。

        返回:
        int: 候选池数据的长度。
        """
        return len(self.cand_pool)

    def construct_messages(self, data_dict):
        """
        根据数据字典构建消息列表。

        参数:
        data_dict (dict): 包含文本和图像信息的数据字典。

        返回:
        list: 包含用户消息和助手消息的列表。
        """
        if 'txt' in data_dict and 'image' in data_dict:
                    message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data_dict['image']},
                    {"type": "text", "text": f"{data_dict['txt']}{self.prompt[0]}"}
                    # 若需要可将注释行代码启用，替换上面的文本内容
                    # {"type": "text", "text": f"{data_dict['txt']}"}
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
                        # 若需要可将注释行代码启用，替换上面的文本内容
                        # {"type": "text", "text": f"{data_dict['txt']}"}
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
                        # 若需要可将注释行代码启用，替换上面的文本内容
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
        """
        根据索引获取候选样本实例。

        参数:
        index (int): 候选样本在候选池数据中的索引。

        返回:
        dict: 包含候选样本信息的字典，如文本、图像、模态、哈希后的 did 等。
        """
        # 从候选池数据中获取指定索引的条目
        mbeir_cand_pool_entry = self.cand_pool[index]
        # 获取候选样本的图像路径
        img_path = mbeir_cand_pool_entry.get("img_path", None)
        # 加载并预处理图像，返回图像的完整路径（若存在）
        img = _load_and_preprocess_image(img_path, self.image_path_prefix)
        # 获取候选样本的 did（唯一标识符）
        did = mbeir_cand_pool_entry.get("did", None)
        # 获取候选样本的文本信息，若为空则返回空字符串
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        # 格式化候选样本的文本，如去除多余空格、换行符，首字母大写等
        cand_txt = format_string(f"{cand_txt}")
        # 获取候选样本的模态信息
        cand_modality = mbeir_cand_pool_entry.get("modality", None)

        # 根据图像和文本的存在情况构建实例字典
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
        # 对 did 进行哈希处理，并添加到实例字典中
        instance.update({"did": hash_did(did)})
        return instance 

    def __getitem__(self, i):
        """
        根据索引获取候选样本的消息和哈希后的 did。

        参数:
        i (int): 候选样本在候选池数据中的索引。

        返回:
        tuple: 包含候选样本消息和哈希后的 did 的元组。
        """
        # 获取候选样本实例
        candidate = self.get_instance(i)
        # 从实例中获取哈希后的 did
        did = candidate['did']
        # 构建候选样本的消息
        candidate_message = self.construct_messages(candidate)
        return candidate_message, did

def _load_data(data_path):
    """
    验证并加载数据。

    参数:
    data_path (str): 数据文件的路径。

    返回:
    list: 加载的数据条目列表。
    """
    # 检查数据文件路径是否存在
    assert os.path.exists(data_path), f"Data Path {data_path} does not exist"
    # 检查数据文件是否为 jsonl 格式
    assert data_path.endswith(".jsonl"), f"Data Path {data_path} is not a jsonl file"
    # 调用 _load_data_jsonl 函数加载 jsonl 文件中的数据
    data_entries = _load_data_jsonl(data_path)
    return data_entries

def _load_query_data(query_data_path):
    """
    加载查询数据。

    参数:
    query_data_path (str): 查询数据文件的路径。

    返回:
    list: 加载的查询数据条目列表。
    """
    # 调用 _load_data 函数加载查询数据
    query_data = _load_data(query_data_path)
    return query_data

def _load_cand_pool_as_dict(cand_pool_data_path):
    """
    加载候选池数据并转换为字典形式。

    参数:
    cand_pool_data_path (str): 候选池数据文件的路径。

    返回:
    dict: 以 did 为键，候选池条目为值的字典。
    """
    # 调用 _load_data 函数加载候选池数据
    cand_pool = _load_data(cand_pool_data_path)
    cand_pool_dict = {}
    for cand_pool_entry in cand_pool:
        # 获取候选池条目的 did
        did = cand_pool_entry.get("did")
        # 确保 did 存在
        assert did, f"Cannot find did for {cand_pool_entry}"
        # 将候选池条目添加到字典中，以 did 为键
        cand_pool_dict[did] = cand_pool_entry
    cand_pool = cand_pool_dict
    return cand_pool 

def _load_query_instructions(instructions_path):
    """
    验证并加载查询指令。

    参数:
    instructions_path (str): 指令文件的路径。

    返回:
    dict: 以数据集 ID、查询模态、候选模态组合为键，提示列表为值的字典。
    """
    # 检查指令文件路径是否存在
    assert os.path.exists(instructions_path), f"Instructions Path {instructions_path} does not exist"
    # 检查指令文件是否为 tsv 格式
    assert instructions_path.endswith(".tsv"), f"Instructions Path {instructions_path} is not a tsv file"
    prompts_dict = {}
    with open(instructions_path, "r") as f:
        # 跳过文件的第一行（通常为表头）
        next(f)  
        for line in f.readlines():
            # 按制表符分割每行数据
            parts = line.strip().split("\t")
            # 构建键，格式为 "数据集 ID, 查询模态, 候选模态"
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            # 过滤掉空的提示，将非空提示存储在列表中
            prompts = [p for p in parts[4:] if p]  
            # 将键值对添加到字典中
            prompts_dict[key] = prompts
    query_instructions = prompts_dict
    return query_instructions 

def _get_random_cand(cand_list):
    """
    从候选列表中随机选择一个候选。

    参数:
    cand_list (list): 候选列表。

    返回:
    随机选择的候选。
    """
    return random.choice(cand_list)

def format_string(s):
    """
    格式化字符串，去除多余空格、换行符，首字母大写，并在必要时添加句点。

    参数:
    s (str): 要格式化的字符串。

    返回:
    str: 格式化后的字符串。
    """
    # 去除字符串中的回车符，去除首尾空格和双引号
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:  # 如果字符串不为空
        # 将字符串的首字母大写
        s = s[0].upper() + s[1:]
        # 如果字符串的最后一个字符不是句号、问号或感叹号，则添加句号
        s = s + "." if s[-1] not in [".", "?", "!"] else s
    return s

def _get_random_query_prompt(dataset_id, query_modality, cand_modality, query_instructions):
    """
    根据数据集 ID、查询模态、候选模态从查询指令中随机选择一个提示。

    参数:
    dataset_id (str): 数据集 ID。
    query_modality (str): 查询模态。
    cand_modality (str): 候选模态。
    query_instructions (dict): 查询指令字典。

    返回:
    str: 随机选择的提示。
    """
    # 构建键，格式为 "数据集 ID, 查询模态, 候选模态"
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    # 从查询指令字典中获取对应的提示列表
    prompts = query_instructions.get(key, [])
    # 确保提示列表不为空
    assert prompts, f"Cannot find prompts for {key}"
    # 从提示列表中随机选择一个提示并格式化
    prompt = format_string(random.choice(prompts))
    # 确保提示不为空
    assert prompt, f"Prompt is empty for {key}"
    return prompt

def _load_and_preprocess_image(query_img_path, image_path_prefix):
    """
    加载并预处理图像。

    参数:
    query_img_path (str): 图像的相对路径。
    image_path_prefix (str): 图像路径的前缀。

    返回:
    str: 图像的完整路径（若存在），否则返回 None。
    """
    if not query_img_path:
        return None
    # 拼接图像的完整路径
    full_query_img_path = os.path.join(image_path_prefix, query_img_path)
    # 确保图像路径存在
    assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
    return full_query_img_path

def _prepare_data_dict(txt, img_path, image_path_prefix):
    """
    准备数据字典，根据文本和图像路径构建包含文本和图像信息的字典。

    参数:
    txt (str): 文本信息。
    img_path (str): 图像的相对路径。
    image_path_prefix (str): 图像路径的前缀。

    返回:
    dict: 包含文本和图像信息的字典。
    """
    # 加载并预处理图像
    img = _load_and_preprocess_image(img_path, image_path_prefix)
    if img is None:
        return {'txt': txt}
    elif txt == '':
        return {'image': img}
    return {"txt": txt, "image": img}

def _load_data_jsonl(datapath):
    """
    加载 jsonl 文件中的数据。

    参数:
    datapath (str): jsonl 文件的路径。

    返回:
    list: 加载的数据条目列表。
    """
    data_entries = []
    with open(datapath, "r") as fin:
        for line in fin:
            # 将每行 json 数据解析为字典并添加到列表中
            data_entry = json.loads(line)
            data_entries.append(data_entry)
    return data_entries

def hash_qid(qid):
    """
    对查询的 qid 进行哈希处理。

    参数:
    qid (str): 查询的 qid，格式为 "数据集 ID:数据内 ID"。

    返回:
    int: 哈希后的 qid。
    """
    # 将 qid 按冒号分割为数据集 ID 和数据内 ID，并转换为整数
    dataset_id, data_within_id = map(int, qid.split(":"))
    # 根据上限计算哈希后的 qid
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id

def unhash_qid(hashed_qid):
    """
    对哈希后的 qid 进行反哈希处理。

    参数:
    hashed_qid (int): 哈希后的 qid。

    返回:
    str: 反哈希后的 qid，格式为 "数据集 ID:数据内 ID"。
    """
    # 计算数据集 ID
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    # 计算数据内 ID
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def hash_did(did):
    """
    对候选样本的 did 进行哈希处理。

    参数:
    did (str): 候选样本的 did，格式为 "数据集 ID:数据内 ID"。

    返回:
    int: 哈希后的 did。
    """
    # 将 did 按冒号分割为数据集 ID 和数据内 ID，并转换为整数
    dataset_id, data_within_id = map(int, did.split(":"))
    # 根据上限计算哈希后的 did
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id

def unhash_did(hashed_did):
    """
    对哈希后的 did 进行反哈希处理。

    参数:
    hashed_did (int): 哈希后的 did。

    返回:
    str: 反哈希后的 did，格式为 "数据集 ID:数据内 ID"。
    """
    # 计算数据集 ID
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    # 计算数据内 ID
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def _load_cand_pool(cand_pool_data_path):
    """
    加载候选池数据。

    参数:
    cand_pool_data_path (str): 候选池数据文件的路径。

    返回:
    list: 加载的候选池数据条目列表。
    """
    # 调用 _load_data 函数加载候选池数据
    cand_pool = _load_data(cand_pool_data_path)
    return cand_pool