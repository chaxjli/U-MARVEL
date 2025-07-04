import json
from transformers import AutoProcessor
from collections import OrderedDict
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
import torch
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
import argparse
from dataset.datasets_mbeir import QueryDataset, CandidateDataset
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000
# 导入自定义的工具函数 debug --------------------------------------------------------------------------------
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import time
# --------------------------------------------------------------------------------------------------------
# 定义数据文件数组
query_data_paths = [
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_mscoco_task0_test.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_mscoco_task3_test.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_cirr_task7_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_fashioniq_task7_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_webqa_task1_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_nights_task4_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_oven_task6_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_infoseek_task6_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_fashion200k_task0_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_fashion200k_task3_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_visualnews_task3_test.jsonl",
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_edis_task2_test.jsonl",                # 4 机 0.8 h
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_visualnews_task0_test.jsonl",          # 4 机 0.4 h
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_webqa_task2_test.jsonl",               # 4 机 0.3 h
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_oven_task8_test.jsonl",                # 4 机 0.3 h
    # "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/query/test/mbeir_infoseek_task8_test.jsonl",            # 4 机 0.5 h

]

query_cand_pool_paths = [
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl"
    # 若所有 query_cand_pool_path 都一样，可简化逻辑
]

cand_pool_paths = [
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_nights_task4_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_edis_task2_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl",

]

qrels_paths = [
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_mscoco_task0_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_cirr_task7_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_fashioniq_task7_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_webqa_task1_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_nights_task4_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_oven_task6_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_infoseek_task6_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_fashion200k_task0_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_fashion200k_task3_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_visualnews_task3_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_edis_task2_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_visualnews_task0_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_webqa_task2_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_oven_task8_test_qrels.txt",
    "/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_infoseek_task8_test_qrels.txt",

]
def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

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
    
    for query_data_path_id in range(len(query_data_paths)):

        rank0_print(f"开始处理第 {query_data_path_id} 个任务","开始时间: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        rank0_print(query_data_paths[query_data_path_id])

        query_data_path = query_data_paths[query_data_path_id]
        query_cand_pool_path = query_cand_pool_paths[0]
        cand_pool_path = cand_pool_paths[query_data_path_id]
        qrels_path = qrels_paths[query_data_path_id]
        rank0_print(f"Processing {query_data_path}")
        filename = os.path.basename(qrels_path)
        result = filename.replace("_qrels.txt", "")
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
        
        query_dataloader = DataLoader(query_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=query_data_collator)
        candidate_dataloader = DataLoader(cand_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=cand_data_collator)
        
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
            model.use_instruction_mask = (args. model_use_instruction_mask == "True")
            query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
            for batch_idx,batch in enumerate(tqdm(query_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                # 处理查询集数据，此处必须将 use_instruction_mask 设置为 还原为原来的值
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
            
            # 处理候选池数据，此处必须将 use_instruction_mask 设置为 False
            model.use_instruction_mask = False
            for batch_idx,batch in enumerate(tqdm(candidate_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                # 开始进行推理 rank0_print("batch",batch)
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
        
        # edis_task2_test.jsonl 会占据很大的显存，所以仅针对这个任务 del model
        # 其他任务不需要释放 model，因为 model 在后续的任务中会被重复使用
        if query_data_path_id == 15:
            model = model.to("cpu")    # 先将模型移出 NPU
            del model
            accelerator.free_memory()  # 关键！释放加速器持有的资源
            torch.npu.empty_cache()    # 清空 NPU 缓存
        
        query_features = torch.cat(query_features, dim=0).to(device)
        candidate_features = torch.cat(candidate_features, dim=0).to(device)
        
        if is_main_process:
            # Adjust the order according to ids 
            import numpy as np 

            index = []
            scores = []
            for i in range(len(query_features)):
                query_feature = query_features[i:i+1]
                score = query_feature @ candidate_features.T # (1, num_candidate)
                topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
                topk_indexes = topk_indexes.squeeze().tolist()
                index.append(topk_indexes)
                scores.append(topk_score.tolist())
            cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
            query_names = [unhash_qid(item) for item in query_ids]
            save_dir_name = args.save_dir_name
            if not os.path.exists(save_dir_name):
                os.makedirs(save_dir_name)
            save_name = qrels_path.split('/')[-1].replace('_qrels.txt', '')
            model_name = args.model_id.split('/')[-1]
            save_name = f"{save_name}_{model_name}"
            with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
                json.dump(query_names, f, indent=2)
            with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
                json.dump(cand_names.tolist(), f, indent=2)
            with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
                json.dump(scores, f, indent=2)
            torch.save(query_features.cpu(), f"{save_dir_name}/{save_name}_query_features.pth")
            torch.save(candidate_features.cpu(), f"{save_dir_name}/{save_name}_candidate_features.pth")
            
            # 释放显存
            del query_features, candidate_features
            accelerator.free_memory()
            torch.npu.empty_cache()

            with open(f"{save_dir_name}/{save_name}_query_ids.json", 'w') as f:
                json.dump(query_ids, f, indent=2)
            with open(f"{save_dir_name}/{save_name}_candidate_ids.json", 'w') as f:
                json.dump(candidate_ids, f, indent=2)
            
            # 释放显存
            del query_ids, candidate_ids
            accelerator.free_memory()
            torch.npu.empty_cache()

            qrel, qid_to_taskid = load_qrel(qrels_path)
            k_lists = [1, 5, 10, 50]
            res = {}
            for k in k_lists:
                res[f'recall_{k}'] = []

            for ind, query_name in enumerate(tqdm(query_names)):
                relevant_docs = qrel[query_name]
                retrieved_indices_for_qid = cand_names[ind]
                for k in k_lists:
                    recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                    res[f'recall_{k}'].append(recall_at_k)
            
            # 释放显存
            del query_names, cand_names
            accelerator.free_memory()
            torch.npu.empty_cache()

            for k in k_lists:
                print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

            model_name = model_id.split('/')[-1]
            with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
                f.write(qrels_path + '\n')
                for k in k_lists:
                    f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')
            
            del qrel, qid_to_taskid
            accelerator.free_memory()
            torch.npu.empty_cache()
        
        torch.npu.empty_cache()
        rank0_print(f"显存释放完成，当前占用: {torch.npu.memory_allocated()/1024**3:.2f} GB")
        rank0_print(f"第 {query_data_path_id} 个任务处理完成: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        rank0_print("*"*50)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_dir_name', type=str)
    parser.add_argument('--image_path_prefix', type=str,default="/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/")
    parser.add_argument('--query_dataset_has_instruction', type=str)
    parser.add_argument('--query_dataset_use_instruction_token', type=str)
    parser.add_argument('--model_mean_pooling', type=str)
    parser.add_argument('--model_use_bi_atten', type=str)
    parser.add_argument('--model_use_latent_atten', type=str)
    parser.add_argument('--model_use_instruction_mask', type=str)
    args = parser.parse_args()
    eval(args)



