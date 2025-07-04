import os 
import numpy as np
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
# from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_github import Qwen2VLRetForConditionalGeneration

import torch 
import argparse
from dataset.datasets_vist import VistDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
import json 
# 导入自定义的工具函数 debug --------------------------------------------------------------------------------
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
# --------------------------------------------------------------------------------------------------------
def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )
    # 重置模型的参数
    model.use_instruction_mask = args.use_instruction_mask == 'True'
    model.use_bi_atten = args.use_bi_atten == 'True'
    model.mean_pooling = args.mean_pooling == 'True'
    model._set_model_use_bi_atten()
    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer
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

    query_dataset = VistDataset(
        data_path=args.data_path, 
        image_path_prefix=args.image_path_prefix,
        type='query'
    )

    cand_dataset = VistDataset(
        data_path=args.data_path,
        image_path_prefix=args.image_path_prefix,
        type='image',
    )

    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    model.eval()
    rank0_print("model.model.use_bi_atten: ", model.model.use_bi_atten)

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
    all_seen_ids = []
    
    from tqdm import tqdm 
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
        # model.use_instruction_mask = True  # 处理查询集数据，此处必须将 use_instruction_mask 设置为 True
        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)
        
        model.use_instruction_mask = False  # 处理候选集数据，此处必须将 use_instruction_mask 设置为 False
        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]

            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)


    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        # Adjust the order according to ids 
        import numpy as np 
        query_ids = np.array(query_ids) 
        sorted_query_indices = np.argsort(query_ids)
        query_features = query_features[sorted_query_indices]

        story_data = list(query_dataset.story_data.values())
        
        index = []
        print(f'len query_features: {len(query_features)}')
        print(f'len candidate_features: {len(candidate_features)}')
        for i in range(len(query_features)):
            query_feature = query_features[i:i+1]
            score = query_feature @ candidate_features.T 
            seen_ids = [s['image_id'] for s in story_data[i][:-1]]
            all_seen_ids.append(seen_ids)
            for seen_id in seen_ids:
                seen_idx = candidate_ids.index(seen_id)
                score[0, seen_idx] = -100
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)

        cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
        query_names = query_ids 

        k_lists = [1, 5, 10]
        res = {}

        for k in k_lists:
            res[f"recall_{k}"] = []

        for ind, query_name in enumerate(tqdm(query_names)):
            relevant_docs = [story_data[ind][-1]['image_id']]
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)

        for k in k_lists:
            print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

        model_name = args.model_id.split('/')[-1]        
        # debug 创建保存结果的目录 ------------------------------------------------------------
        save_dir_name = args.save_dir_name
        print("save_dir_name: ", save_dir_name)
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name, exist_ok=True)  # 递归创建目录，若存在则忽略
        # ----------------------------------------------------------------------------------
        with open(f"{save_dir_name}/{model_name}.txt", 'w') as f:
            f.write('vist evaluation' + '\n')
            for key in res:
                f.write(f"{key} = {sum(res[key]) / len(res[key])}" + "\n")

        if args.save_for_rerank:
            save_for_rerank(query_features, candidate_features, query_ids, candidate_ids, all_seen_ids, save_dir_name)

def save_for_rerank(query_features, candidate_features, query_ids, candidate_ids, all_seen_ids, save_dir_name):
    index = []
    scores = []
    for i in range(len(query_features)):
        score = query_features[i : i + 1] @ candidate_features.T 
        for seen_id in all_seen_ids[i]:
            seen_idx = candidate_ids.index(seen_id)
            score[0, seen_idx] = -100
        topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        index.append(topk_indexes)
        scores.append(topk_score.tolist())
    
    cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
    query_names = query_ids 
   
    with open(f"{save_dir_name}/vist_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/vist_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/vist_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_for_rerank', action='store_true')
    parser.add_argument('--save_dir_name', type=str)
    parser.add_argument('--use_instruction_mask', type=str, default='True', help='Whether to use instruction mask')
    parser.add_argument('--use_bi_atten', type=str, default='True', help='Whether to use bi attention')
    parser.add_argument('--mean_pooling', type=str, default='True', help='Whether to use mean pooling')
    args = parser.parse_args()
    eval(args)