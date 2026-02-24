import json
from transformers import AutoProcessor
from collections import OrderedDict,defaultdict
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
import torch
# 不确定是否需要适配
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
import argparse
from dataset.datasets_mmeb_multi_gpu import QueryDataset, CandidateDataset
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
import numpy as np
from tqdm import tqdm
import collections
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000
# 导入自定义的工具函数 debug --------------------------------------------------------------------------------
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import time
mmeb_result = {}
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
subsets = Overall[:]  # 这里的 Overall 是一个列表，包含了所有的子集名称
# subsets = ["Place365",]
# --------------------------------------------------------------------------------------------------------
# 定义数据文件数组
# ./data/MMEB-eval/A-OKVQA/A-OKVQA_query.jsonl
# ./data/MMEB-eval/A-OKVQA/A-OKVQA_candidate.jsonl
# ./data/MMEB-eval/A-OKVQA/A-OKVQA_qrel.json
# ./data/MMEB-eval/mmeb_union_candidate.jsonl
# ./data/MMEB-eval/mmeb_union_query.jsonl
query_data_paths = [f"./data/MMEB-eval/{subset}/{subset}_query.jsonl" for subset in subsets]
query_cand_pool_paths = ["./data/MMEB-eval/mmeb_union_candidate.jsonl"]
cand_pool_paths = [f"./data/MMEB-eval/{subset}/{subset}_candidate.jsonl" for subset in subsets]
qrels_paths = [f"./data/MMEB-eval/{subset}/{subset}_qrel.json" for subset in subsets]

def get_pred(qry_t, tgt_t, normalization=False):
    """
    计算查询向量与目标向量的相似度并获取预测结果
    参数:
        qry_t (np.ndarray): 查询向量（形状为 [dim,]）
        tgt_t (np.ndarray): 目标向量集合（形状为 [num_candidates, dim]）
        normalization (bool): 是否对向量进行归一化处理（默认: False）
    
    返回:
        scores (np.ndarray): 相似度得分数组（形状为 [num_candidates,]）
        pred (int): 预测的最佳匹配索引（得分最高的目标向量索引）
    
    数学原理:
        - 若 normalization=True:
          相似度计算为余弦相似度，公式为:
          score = (tgt_t · qry_t) / (||tgt_t|| × ||qry_t||)
          其中 ||·|| 表示L2范数，点积结果除以两个向量的范数乘积，结果范围为 [-1, 1]。
        
        - 若 normalization=False:
          相似度计算为点积，公式为:
          score = tgt_t · qry_t
          结果范围取决于向量的模长，可能为任意实数。
    """
    if normalization:
        # 计算查询向量的L2范数
        qry_t_norm = np.linalg.norm(qry_t)
        # 计算所有目标向量的L2范数（沿维度1，即每个样本的范数）
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        # 计算点积并除以范数乘积（余弦相似度）
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        # 直接计算点积作为相似度得分
        scores = np.dot(tgt_t, qry_t)
    # 获取得分最高的目标向量索引（预测结果）
    pred = np.argmax(scores)
    return scores, pred

def get_cand_info(cand_feature_path, cand_ids_path):
    try:
        cand_features = torch.load(cand_feature_path)
        with open(cand_ids_path, 'r') as f:
            cand_ids = json.load(f)
        cand_ids = [unhash_did(item) for item in cand_ids]
        return cand_features, cand_ids
    except Exception as e:
        print(str(e))
        return None, None

def get_query_info(query_feature_path, query_names_path):
    query_features = torch.load(query_feature_path)
    with open(query_names_path, 'r') as f:
        query_names = json.load(f)
    return query_features, query_names

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    # filename 是一个 json 字典
    qrel = collections.defaultdict(list)  # 使用 defaultdict 来简化代码
    with open(filename, "r") as f:
        data = json.load(f)
    for query_id, doc_id in data.items():
        qrel[query_id].extend(doc_id)
    for query_id in qrel.keys():
        qrel[query_id] = list(set(qrel[query_id]))  # 去重
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}")
    return qrel

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents
    # Get the set of indices for the top k retrieved documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    # Convert the relevant documents to a set
    relevant_docs_set = set(relevant_docs)
    # Check if there is an intersection between relevant docs and top k retrieved docs
    # If there is, we return 1, indicating successful retrieval; otherwise, we return 0
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0
def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )
    # 处理模型的配置项--------------------------------------------------------------------------------
    model.mean_pooling = args.model_mean_pooling == "True"
    model.use_bi_atten = args.model_use_bi_atten == "True"
    model.use_latent_atten = args.model_use_latent_atten == "True"
    model.use_instruction_mask = args.model_use_instruction_mask == "True"
    # 加载 latent_atten 模块--------------------------------------------------------------------------
    latent_atten_path = os.path.join(model_id, "latent_atten.bin")
    if os.path.isfile(latent_atten_path):
        assert model.use_latent_atten == True, "模型配置文件中 use_latent_atten 为 False，但是路径当中存在 latent_atten 模型"
        latent_atten_state_dict = OrderedDict()
        ori_ckpts = torch.load(latent_atten_path)
        for k, v in ori_ckpts.items():
            # 这里的 base_model.model. 是因为在保存的时候，模型的 key 值是这样的
            latent_atten_state_dict[k.replace("base_model.model.", "")] = v
        model.load_state_dict(latent_atten_state_dict, strict=False)
    # ------------------------------------------------------------------------------------------------

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer 
    tokenizer.model_max_length = args.model_max_length
    
    # 为每个 token 创建独立配置项 --------------------------------------------------------------------------------
    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens
        model.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(emb_token)
        if emb_token == "<instruction_start>":
            model.config.instruction_start_token_id = token_id
        elif emb_token == "<instruction_end>":
            model.config.instruction_end_token_id = token_id
        else:
            model.config.emb_token_id = token_id  # 默认通用 token
    add_embed_token(tokenizer, model)
    if model.use_instruction_mask: # 这里暂时先这样处理，后续再优化
        add_embed_token(tokenizer, model, emb_token="<instruction_start>")
        add_embed_token(tokenizer, model, emb_token="<instruction_end>")
    
    # ------------------------------------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process
    rank0_print( "use_latent_atten: ",model.use_latent_atten,type(model.use_latent_atten))
    for name, param in model.named_parameters():
        if "temp" in name:
            rank0_print("温度参数:", name)
            rank0_print("温度参数的值:", param)
    
    for query_data_path_id in range(len(subsets)):
        subset = subsets[query_data_path_id]
        rank0_print(f"开始处理第 {query_data_path_id} 个任务","开始时间: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        rank0_print(query_data_paths[query_data_path_id])
        query_data_path = query_data_paths[query_data_path_id]
        query_cand_pool_path = query_cand_pool_paths[0]
        cand_pool_path = cand_pool_paths[query_data_path_id]
        qrels_path = qrels_paths[query_data_path_id]
        rank0_print(f"Processing {query_data_path}")
        filename = os.path.basename(qrels_path)
        result = filename.replace("_qrel.json", "")
        # 构建要检查的文件路径
        MODEL_NAME = args.model_id.split('/')[-1]
        check_file = os.path.join(args.save_dir_name, f"{result}_{MODEL_NAME}_candidate_features.pth")
        rank0_print(f"当前显存占用: {torch.npu.memory_allocated()/1024**3:.2f} GB")
        # 打印 check_file 路径用于调试
        rank0_print(f"正在检查文件: {check_file}")
        if os.path.isfile(check_file):
            rank0_print(f"文件 {check_file} 存在，跳过本次任务。")
            rank0_print("*"*50)
            continue
        query_dataset = QueryDataset(
            query_data_path=query_data_path, 
            cand_pool_path=query_cand_pool_path,
            instructions_path=args.instructions_path,
            image_path_prefix=args.image_path_prefix,
            use_instruction_token=(args.query_dataset_use_instruction_token == "True"),       # 数据集是否使用指令 token
            has_instruction= (args.query_dataset_has_instruction == "True"),                  # 数据集是否有指令
        )
        cand_dataset = CandidateDataset(
            query_data_path=query_data_path, 
            cand_pool_path=cand_pool_path,
            instructions_path=args.instructions_path,
            image_path_prefix=args.image_path_prefix,
        )
        query_data_collator = MbeirQueryDataCollator(tokenizer=tokenizer, processor=processor, \
                                                    has_instruction=query_dataset.has_instruction, \
                                                    use_instruction_token=query_dataset.use_instruction_token)
        cand_data_collator = MbeirCandidateDataCollator(tokenizer=tokenizer, processor=processor)
        
        # batch_size 原来的值是 16
        query_dataloader = DataLoader(query_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=query_data_collator)
        candidate_dataloader = DataLoader(cand_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=cand_data_collator)
        model = model.to(device)
        # 打印模型的信息 --------------------------------------------------------------------------------
        # query_data_collator 的信息 --------------------------------------------------------------------------------
        rank0_print("query_data_collator 的 has_instruction 是：",query_data_collator.has_instruction)
        rank0_print("query_data_collator 的 use_instruction_token 是：",query_data_collator.use_instruction_token)
        # 打印 query_dataset 的信息 --------------------------------------------------------------------------------
        rank0_print("query_dataset 的长度是：",len(query_dataset))  
        rank0_print("query_dataset 的咒语是：",query_dataset.prompt)
        rank0_print("cand_dataset 使用的咒语： ",cand_dataset.prompt)
        rank0_print("query_dataset 是否使用指令是：",query_dataset.has_instruction)
        rank0_print("query_dataset 是否使用指令 token 是：",query_dataset.use_instruction_token)
        rank0_print("模型初始化完成 ————————————————————————————————————————————————————————————————————————")
        rank0_print("mean_pooling: ",model.mean_pooling ,"use_bi_atten: ",model.use_bi_atten)
        rank0_print( "use_latent_atten: ",model.use_latent_atten)
        rank0_print("use_instruction_mask: ",model.use_instruction_mask,type(model.use_instruction_mask))
        rank0_print("模型的类名是：",model.__class__.__name__)
        rank0_print("脚本的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        rank0_print("脚本的参数是: ",args)
        # ------------------------------------------------------------------------------------------------
        model.eval()
        def tensors_to_device(data, device, dtype=model.dtype):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if key == 'pixel_values':
                        data[key] = data[key].to(device).to(dtype)
                    else:
                        data[key] = data[key].to(device)
            return data 

        query_features = []
        query_ids = []
        candidate_features = []
        candidate_ids = []
        from tqdm import tqdm 
        with torch.no_grad():
            candidate_batch_times = 0  # 收集候选集前 10 个batch 的时间
            query_batch_times = 0      # 收集查询集前 10 个batch 的时间
            query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
            for batch_idx,batch in enumerate(tqdm(query_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                # 处理查询集数据，此处必须将 use_instruction_mask 设置为 还原为原来的值
                model.use_instruction_mask = (args. model_use_instruction_mask == "True")
                query_embed, batch_query_ids, _ = model(**batch, inference=True)
                query_embed = F.normalize(query_embed, dim=-1)
                query_embed = accelerator.gather_for_metrics(query_embed)
                batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
                query_ids.extend(batch_query_ids)
                query_features.append(query_embed.cpu())  # 替换原append语句
                if batch_idx == 99:
                    query_batch_times = time.time() - start_time
                    rank0_print(f"查询集前 100 个batch的时间: {query_batch_times/60.00} min")
                    query_batch_times = query_batch_times * len(query_dataloader) / 100
                    rank0_print(f"查询集所有 batch的时间: {query_batch_times/3600.00} h")

            for batch_idx,batch in enumerate(tqdm(candidate_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                # 开始进行推理 rank0_print("batch",batch)
                # 处理候选池数据，此处必须将 use_instruction_mask 设置为 False
                model.use_instruction_mask = False
                candidate_embed, _, batch_candidate_ids = model(**batch, inference=True)
                candidate_embed = F.normalize(candidate_embed, dim=-1)
                candidate_embed = accelerator.gather_for_metrics(candidate_embed)
                batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
                candidate_ids.extend(batch_candidate_ids)
                candidate_features.append(candidate_embed.cpu())  # 替换原append语句
                if batch_idx == 99:
                    candidate_batch_times = time.time() - start_time
                    rank0_print(f"候选集前 100 个 batch 的时间: {candidate_batch_times/60.00} min")
                    candidate_batch_times = candidate_batch_times * len(candidate_dataloader) / 100
                    rank0_print(f"候选集所有 batch 的时间: {candidate_batch_times/3600.00} h")

        query_features = torch.cat(query_features, dim=0).to(device)
        candidate_features = torch.cat(candidate_features, dim=0).to(device)
        rank0_print(f"查询特征的形状: {query_features.shape}, 候选特征的形状: {candidate_features.shape}")
        save_dir_name = args.save_dir_name
        if is_main_process:
            rank0_print("is_main_process:", is_main_process)
            # Adjust the order according to ids 
            index = []
            scores = []
            for i in range(len(query_features)):
                query_feature = query_features[i:i+1]
                score = query_feature @ candidate_features.T # (1, num_candidate)
                topk_score, topk_indexes = torch.topk(score, k=2, dim=-1)
                topk_indexes = topk_indexes.squeeze().tolist()
                index.append(topk_indexes)
                scores.append(topk_score.tolist())
            cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
            query_names = [unhash_qid(item) for item in query_ids]
            
            if not os.path.exists(save_dir_name):
                os.makedirs(save_dir_name)
            save_name = qrels_path.split('/')[-1].replace('_qrel.json', '')
            model_name = args.model_id.split('/')[-1]
            save_name = f"{save_name}_{model_name}"
            with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
                json.dump(query_names, f, indent=2)
            torch.save(query_features.cpu(), f"{save_dir_name}/{save_name}_query_features.pth")
            torch.save(candidate_features.cpu(), f"{save_dir_name}/{save_name}_candidate_features.pth")
            with open(f"{save_dir_name}/{save_name}_query_ids.json", 'w') as f:
                json.dump(query_ids, f, indent=2)
            with open(f"{save_dir_name}/{save_name}_candidate_ids.json", 'w') as f:
                json.dump(candidate_ids, f, indent=2)
            print("#"*100)
            # 从这里开始处理 mmeb 数据集 --------------------------------------------------------------------------------
            query_feature_path, query_names_path = f"{save_dir_name}/{save_name}_query_features.pth", f"{save_dir_name}/{save_name}_query_names.json"
            cand_feature_path, cand_ids_path = f"{save_dir_name}/{save_name}_candidate_features.pth", f"{save_dir_name}/{save_name}_candidate_ids.json"
            qry_tensor, qry_index = get_query_info(query_feature_path, query_names_path)
            tgt_tensor, tgt_index = get_cand_info(cand_feature_path, cand_ids_path)
            # 检查保存和加载的特征是否一致
            candidate_ids = [unhash_did(item) for item in candidate_ids]
            assert torch.equal(qry_tensor, query_features.cpu()), "查询特征不匹配"
            assert torch.equal(tgt_tensor, candidate_features.cpu()), "候选特征不匹配"
            assert qry_index == query_names, "查询名称不匹配"
            assert tgt_index == candidate_ids, "候选 ID 不匹配"
            # 释放显存
            del query_ids, candidate_ids,query_features, candidate_features,query_names, cand_names
            accelerator.free_memory()
            torch.npu.empty_cache()
            qrel = load_qrel(qrels_path)
            
            # 构建查询和目标的映射字典
            qry_dict, tgt_dict = {}, {}
            for qry_t, tt in zip(qry_tensor, qry_index):
                qry_dict[tt] = qry_t  # 键: (文本, 图像路径), 值: 编码向量
            rank0_print(f"qry_dict 的长度: {len(qry_dict)}")
            rank0_print(f"qry_dict 的示例: {list(qry_dict.items())[:1]}")
            for tgt_t, tt in zip(tgt_tensor, tgt_index):
                tgt_dict[tt] = tgt_t
            rank0_print(f"tgt_dict 的长度: {len(tgt_index)}")
            rank0_print(f"tgt_dict 的示例: {list(tgt_dict.items())[:1]}")
            # ------------------------------------------------------------------------------------------------
            n_correct = 0  # 正确预测数
            all_pred = []  # 所有预测结果
            with open(query_data_path, "r") as f:
                eval_data = [json.loads(line) for line in f.readlines() if line.strip()]
            all_cand_names = defaultdict(list)  # 保存所有候选名称
            all_cand_scores = defaultdict(list)  # 保存所有候选分数
            for row in eval_data:
                # 获取当前查询样本的编码向量
                qry_t = qry_dict[row["qid"]]  # (dim,)
                # 获取所有候选目标的编码向量
                tgt_t, all_candidates = [], []
                for tt in row["candidates"]:
                    tgt_t.append(tgt_dict[tt])
                    all_candidates.append(tt)
                tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
                scores, pred = get_pred(qry_t, tgt_t, normalization=True)
                # 对分数进行降序排序，获取排序后的索引
                sorted_indices = np.argsort(scores)[::-1]
                sorted_candidates = [row["candidates"][i] for i in sorted_indices]
                sorted_scores = [scores[i] for i in sorted_indices]
                all_cand_names[row["qid"]].extend(sorted_candidates)  # 保存排序后的候选名称
                all_cand_scores[row["qid"]].extend(sorted_scores)    # 保存排序后的分数
                assert sorted_candidates[0] == all_candidates[pred], "预测的候选名称与排序后的候选名称不一致"
                assert all_candidates[0] in qrel[row["qid"]], "gt 候选名称不在 qrel 中"
                # 判断预测是否正确（假设第一个候选为正确答案）
                if pred == 0:
                    n_correct += 1
                all_pred.append(all_candidates[pred])  # 保存预测结果
            # 保存预测结果
            all_cand_names = [all_cand_names[item] for item in qry_index]
            all_cand_scores = [all_cand_scores[item] for item in qry_index]
            all_cand_scores = [[float(x) for x in row] for row in all_cand_scores]
            with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
                json.dump(all_cand_names, f, indent=2)
            with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
                json.dump(all_cand_scores, f, indent=2)
            with open(os.path.join(save_dir_name, f"{subset}_pred.txt"), "w") as f:
                for item in all_pred:
                    f.write(f"{item}\n")
            # 计算并保存评估分数
            score_path = os.path.join(save_dir_name, f"{subset}_score.json")
            rank0_print(f"Outputting final score to: {score_path}")
            with open(score_path, "w") as f:
                score_dict = {
                    "acc": n_correct/len(eval_data),  # 准确率
                    "num_correct": n_correct,         # 正确预测数
                    "num_pred": len(eval_data)        # 总预测数
                }
                json.dump(score_dict, f, indent=4)  # 保存为JSON格式
            rank0_print(f"{subset} accuracy: {n_correct/len(eval_data)}")  # 打印准确率
            rank0_print(f"{subset} num_correct: {n_correct}, num_pred: {len(eval_data)}")  # 打印正确预测数和总预测数
            rank0_print(f"Scores saved to: {score_path}")
            rank0_print("*"*100)
            mmeb_result[subset] = n_correct/len(eval_data)
    # ------------------------------------------------------------------------------------------------
    # 计算 Classification、VQA、Retrieval、Visual Grounding、IND、OOD、Overall 的平均准确率
    if is_main_process:
        Classification_acc = np.mean([mmeb_result[task] for task in Classification if task in mmeb_result])
        VQA_acc = np.mean([mmeb_result[task] for task in VQA if task in mmeb_result])
        Retrieval_acc = np.mean([mmeb_result[task] for task in Retrieval if task in mmeb_result])
        Visual_Grounding_acc = np.mean([mmeb_result[task] for task in Visual_Grounding if task in mmeb_result])
        IND_acc = np.mean([mmeb_result[task] for task in IND if task in mmeb_result])
        OOD_acc = np.mean([mmeb_result[task] for task in OOD if task in mmeb_result])
        Overall_acc = np.mean([mmeb_result[task] for task in Overall if task in mmeb_result])
        # 保存平均准确率
        rank0_print("开始计算各个任务的平均准确率")
        mmeb_result["Classification"] = Classification_acc
        mmeb_result["VQA"] = VQA_acc
        mmeb_result["Retrieval"] = Retrieval_acc
        mmeb_result["Visual_Grounding"] = Visual_Grounding_acc
        mmeb_result["IND"] = IND_acc
        mmeb_result["OOD"] = OOD_acc
        mmeb_result["Overall"] = Overall_acc
        mmeb_result["V1-Overall"] = (10 * Classification_acc + 10 * VQA_acc + 12 * Retrieval_acc + 4 * Visual_Grounding_acc) / 36
        rank0_print(f"Classification accuracy: {Classification_acc}")
        rank0_print(f"VQA accuracy: {VQA_acc}")
        rank0_print(f"Retrieval accuracy: {Retrieval_acc}")
        rank0_print(f"Visual Grounding accuracy: {Visual_Grounding_acc}")
        rank0_print(f"IND accuracy: {IND_acc}")
        rank0_print(f"OOD accuracy: {OOD_acc}")
        rank0_print(f"Overall accuracy: {Overall_acc}")
        rank0_print("各个任务的平均准确率计算完成，且平均准确率已保存到结果字典中")
        # ------------------------------------------------------------------------------------------------    
        # 保存所有结果
        save_dir_name = args.save_dir_name
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        result_path = os.path.join(save_dir_name, "mmeb_eval_results.json")
        with open(result_path, "w") as f:
            json.dump(mmeb_result, f, indent=4)
        rank0_print(f"所有评估结果已保存到: {result_path}")
        rank0_print("脚本结束的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_dir_name', type=str)
    parser.add_argument('--image_path_prefix', type=str,default="./data/MMEB-eval/eval_images/")
    parser.add_argument('--query_dataset_has_instruction', type=str)
    parser.add_argument('--query_dataset_use_instruction_token', type=str)
    parser.add_argument('--model_mean_pooling', type=str)
    parser.add_argument('--model_use_bi_atten', type=str)
    parser.add_argument('--model_use_latent_atten', type=str)
    parser.add_argument('--model_use_instruction_mask', type=str)
    args = parser.parse_args()
    eval(args)



