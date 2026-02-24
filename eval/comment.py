def load_qrel(filename):
    """
    qrel: 保存每个查询ID对应的相关文档ID列表（格式：{query_id: [doc_id1, docdoc_id2,...]})
    qid_to_taskid: 保存查询ID到任务ID的映射（格式：{query_id: task_id}）
    """
    qrel = {}
    qid_to_taskid = {}

    with open(filename, "r") as f:
        for line in f:
            # 解析每行数据（假设格式为TREC qrel标准格式的扩展版）
            # 标准格式：<query_id> <iter> <doc_id> <relevance_score>
            # 扩展格式：<query_id> <iter> <doc_id> <relevance_score> <task_id>
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            
            # 仅处理相关度分数>0的文档（过滤不相关文档）
            if int(relevance_score) > 0:  
                # 将文档ID添加到对应查询的列表中
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                
                # 记录查询ID对应的任务ID（每个查询ID只会记录第一次出现的task_id）
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    
    # 打印统计信息
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

# 计算在给定 k 值下的召回率
def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    """
    Args:
        relevant_docs (list):     qrel[query_name], 根据 qrel 得到的 did 列表
        retrieved_indices (list): cand_names[ind],模型检索到的 did 列表，根据相似度得分排序
        k (int): topk 指标
    """
    if not relevant_docs:
        return 0.0 # 如果没有相关文档，召回率为 0
    # 获取前 k 个检索到的文档的索引集合
    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    # 将相关文档转换为集合
    relevant_docs_set = set(relevant_docs)
    # 检查相关文档集合和前 k 个检索到的文档集合是否有交集
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

import json
from transformers import AutoProcessor
import sys 
import os 
# 获取当前文件的绝对路径
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 计算上级目录的路径
module_path = os.path.join(current_file_path, "../")
# 将上级目录添加到系统路径中，以便导入上级目录中的模块
sys.path.append(module_path)
# 导入预训练模型类，这里可能有问题，评估可能需要使用微调后的模型
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration 
# 导入微调后的模型类
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
import torch
# 导入华为 NPU 相关的库，用于适配 NPU 设备
import torch_npu                             
from torch_npu.contrib import transfer_to_npu 
import argparse
# 导入查询数据集类
from dataset.datasets_mbeir import QueryDataset, CandidateDataset
# 导入数据整理器类
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
# 导入分布式训练加速库
from accelerate import Accelerator
import accelerate
# 定义查询数据集数量上限
DATASET_QUERY_NUM_UPPER_BOUND = 500000
# 定义候选数据集数量上限
DATASET_CAN_NUM_UPPER_BOUND = 10000000
# 导入自定义的工具函数
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import time

# 将哈希后的查询 ID 解码为原始 ID
def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

# 将哈希后的文档 ID 解码为原始 ID
def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"




# 评估函数，用于执行整个评估流程
def eval(args):
    # 原始模型的 ID
    original_model_id = args.original_model_id
    # 要使用的模型的 ID
    model_id = args.model_id 
    # 从预训练模型中加载微调后的模型
    model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
    )
    # 处理模型的配置项
    model.mean_pooling = args.model_mean_pooling == "True"
    model.use_bi_atten = args.model_use_bi_atten == "True"
    model.use_latent_atten = args.model_use_latent_atten == "True"
    model.use_instruction_mask = args.model_use_instruction_mask == "True"
    print( "use_latent_atten: ",model.use_latent_atten,type(model.use_latent_atten))

    # 从原始模型仓库中加载处理器
    processor = AutoProcessor.from_pretrained(original_model_id)
    # 获取分词器
    tokenizer = processor.tokenizer 
    # 设置分词器的最大长度
    tokenizer.model_max_length = args.model_max_length
    
    # 为每个 token 创建独立配置项
    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        # 向分词器中添加新的 token
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens
        # 调整模型的词嵌入层大小
        model.resize_token_embeddings(len(tokenizer))
        # 获取新 token 的 ID
        token_id = tokenizer.convert_tokens_to_ids(emb_token)
        if emb_token == "<instruction_start>":
            model.config.instruction_start_token_id = token_id
        elif emb_token == "<instruction_end>":
            model.config.instruction_end_token_id = token_id
        else:
            model.config.emb_token_id = token_id  # 默认通用 token
    # 添加通用 token
    add_embed_token(tokenizer, model)
    # 添加指令开始 token
    add_embed_token(tokenizer, model, emb_token="<instruction_start>")
    # 添加指令结束 token
    add_embed_token(tokenizer, model, emb_token="<instruction_end>")

    # 创建查询数据集对象
    query_dataset = QueryDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix,
        use_instruction_token=(args.query_dataset_use_instruction_token == "True"),       # 数据集是否使用指令 token
        has_instruction= (args.query_dataset_has_instruction == "True"),      # 数据集是否有指令
    )

    # 创建候选数据集对象
    cand_dataset = CandidateDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix                     #  debug
    )

    # 创建查询数据整理器对象
    query_data_collator = MbeirQueryDataCollator(tokenizer=tokenizer, processor=processor, \
                                                has_instruction=query_dataset.has_instruction, \
                                                use_instruction_token=query_dataset.use_instruction_token)
    # 创建候选数据整理器对象
    cand_data_collator = MbeirCandidateDataCollator(tokenizer=tokenizer, processor=processor)
    
    # 创建查询数据加载器
    query_dataloader = DataLoader(query_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    # 创建候选数据加载器
    candidate_dataloader = DataLoader(cand_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    # 初始化分布式训练加速器
    accelerator = Accelerator(mixed_precision='bf16')
    # 获取当前设备
    device = accelerator.device 
    # 判断当前进程是否为主进程
    is_main_process = accelerator.is_main_process
    
    # 打印模型的信息
    # 打印查询数据整理器的信息
    rank0_print("query_data_collator 的 has_instruction 是：",query_data_collator.has_instruction)
    rank0_print("query_data_collator 的 use_instruction_token 是：",query_data_collator.use_instruction_token)
    # 打印查询数据集的信息
    rank0_print("query_dataset 的长度是：",len(query_dataset))  
    rank0_print("query_dataset 的咒语是：",query_dataset.prompt)
    rank0_print("cand_dataset 使用的咒语： ",cand_dataset.prompt)
    rank0_print("query_dataset 是否使用指令是：",query_dataset.has_instruction)
    rank0_print("query_dataset 是否使用指令 token 是：",query_dataset.use_instruction_token)
    rank0_print("模型初始化完成————————————————————————————————————————————————————————————————————————")
    rank0_print("mean_pooling: ",model.mean_pooling ,"use_bi_atten: ",model.use_bi_atten)
    rank0_print( "use_latent_atten: ",model.use_latent_atten)
    rank0_print("use_instruction_mask: ",model.use_instruction_mask,type(model.use_instruction_mask))
    rank0_print("模型的类名是：",model.__class__.__name__)
    rank0_print("脚本的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rank0_print("脚本的参数是: ",args)

    # 将模型设置为评估模式
    model.eval()

    # 将数据中的张量移动到指定设备上
    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    # 存储查询特征
    query_features = []
    # 存储查询 ID
    query_ids = []
    # 存储候选特征
    candidate_features = []
    # 存储候选 ID
    candidate_ids = []

    from tqdm import tqdm 
    with torch.no_grad():
        # 收集候选集前 100 个 batch 的时间
        candidate_batch_times = 0 
        # 收集查询集前 100 个 batch 的时间
        query_batch_times = 0     
        # 使用加速器准备数据加载器和模型
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)

        for batch_idx,batch in enumerate(tqdm(query_dataloader, disable=not is_main_process)):
            if batch_idx == 0:
                start_time = time.time()
                rank0_print("1"*50)
            # 将数据移动到指定设备上
            batch = tensors_to_device(batch, device)
            # 处理查询集数据，此处必须将 use_instruction_mask 设置为 还原为原来的值
            model.use_instruction_mask = (args. model_use_instruction_mask == "True")
            # 进行推理，获取查询嵌入、查询 ID
            query_embed, batch_query_ids, _ = model(**batch, inference=True)
            # 对查询嵌入进行归一化处理
            query_embed = F.normalize(query_embed, dim=-1)
            # 收集查询嵌入
            query_embed = accelerator.gather_for_metrics(query_embed)
            # 收集查询 ID
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed.cpu())  # 替换原append语句
            if batch_idx == 99:
                query_batch_times = time.time() - start_time
                print(f"查询集前 100 个batch的时间: {query_batch_times/60.00} min")
                query_batch_times = query_batch_times * len(query_dataloader) / 100
                print(f"查询集所有 batch的时间: {query_batch_times/3600.00} h")

        for batch_idx,batch in enumerate(tqdm(candidate_dataloader, disable=not is_main_process)):
            if batch_idx == 0:
                start_time = time.time()
            batch = tensors_to_device(batch, device)
            model.use_instruction_mask = False
            candidate_embed, _, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed.cpu())  # 替换原append语句
            if batch_idx == 99:
                candidate_batch_times = time.time() - start_time
                print(f"候选集前 100 个 batch 的时间: {candidate_batch_times/60.00} min")
                candidate_batch_times = candidate_batch_times * len(candidate_dataloader) / 100
                print(f"候选集所有 batch 的时间: {candidate_batch_times/3600.00} h")

    model = model.to("cpu")  
    del model
    accelerator.free_memory()  
    torch.npu.empty_cache()
    query_features = torch.cat(query_features, dim=0).to(device)
    candidate_features = torch.cat(candidate_features, dim=0).to(device)
    
    if is_main_process:
        # Adjust the order according to ids 
        import numpy as np 

        index = []
        scores = []
        for i in range(len(query_features)):
            # 获取当前查询的特征
            query_feature = query_features[i:i+1]
            # 计算查询特征与候选特征的相似度得分
            score = query_feature @ candidate_features.T # (1, num_candidate)
            # 获取前 k 个得分和对应的索引
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)
            scores.append(topk_score.tolist())

        # 将候选 ID 解码为原始 ID
        cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
        # 将查询 ID 解码为原始 ID
        query_names = [unhash_qid(item) for item in query_ids]

        # 保存结果的目录名
        save_dir_name = args.save_dir_name
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        # 保存文件名
        save_name = args.qrels_path.split('/')[-1].replace('_qrels.txt', '')
        model_name = args.model_id.split('/')[-1]
        save_name = f"{save_name}_{model_name}"
        # 保存查询名称
        with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
            json.dump(query_names, f, indent=2)
        # 保存候选名称
        with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
            json.dump(cand_names.tolist(), f, indent=2)
        # 保存相似度得分
        with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
            json.dump(scores, f, indent=2)
        # 保存查询特征
        torch.save(query_features.cpu(), f"{save_dir_name}/{save_name}_query_features.pth")
        # 保存候选特征
        torch.save(candidate_features.cpu(), f"{save_dir_name}/{save_name}_candidate_features.pth")
        # 保存查询 ID
        with open(f"{save_dir_name}/{save_name}_query_ids.json", 'w') as f:
            json.dump(query_ids, f, indent=2)
        # 保存候选 ID
        with open(f"{save_dir_name}/{save_name}_candidate_ids.json", 'w') as f:
            json.dump(candidate_ids, f, indent=2)

        # 加载查询与相关文档的关联文件
        qrel, qid_to_taskid = load_qrel(args.qrels_path)

        # 定义要计算的 k 值列表
        k_lists = [1, 5, 10, 50]
        res = {}

        for k in k_lists:
            res[f'recall_{k}'] = []

        for ind, query_name in enumerate(tqdm(query_names)):
            # 获取当前查询的相关文档
            relevant_docs = qrel[query_name]
            # 获取当前查询检索到的文档索引
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                # 计算当前 k 值下的召回率
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)

        for k in k_lists:
            print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

        model_name = model_id.split('/')[-1]
        with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
            f.write(args.qrels_path + '\n')
            for k in k_lists:
                f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')
