import os
import torch 
# 不确定是否需要适配
# import torch_npu                              # 适配 npu
# from torch_npu.contrib import transfer_to_npu # 适配 npu
import json  
from tqdm import tqdm 
import numpy as np 
import faiss 
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='评估 MBEIR 全局调试脚本')
# 添加参数
parser.add_argument('--model_name', type=str, help='输入文件的路径')
# 解析参数
args = parser.parse_args()
MODEL_NAME = args.model_name
print(MODEL_NAME)
save_dir_name = f"./result/{MODEL_NAME}_eval_results_finetune/global"
if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)
print("保存处理结果文件的目录: ",save_dir_name)
    
cand_feature_path_format = "./result/{}_eval_results_finetune/local/mbeir_{}_test_{}_candidate_features.pth"
cand_ids_path_format = "./result/{}_eval_results_finetune/local/mbeir_{}_test_{}_candidate_ids.json"


query_feature_path_format = "./result/{}_eval_results_finetune/local/mbeir_{}_test_{}_query_features.pth"
query_names_path_format = "./result/{}_eval_results_finetune/local/mbeir_{}_test_{}_query_names.json"

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000
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

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def get_cand_info(task_name):
    try:
        print("开始读取文件--------：  ",task_name)
        cand_features = torch.load(cand_feature_path_format.format(MODEL_NAME,task_name,MODEL_NAME))
        cand_ids = json.load(open(cand_ids_path_format.format(MODEL_NAME,task_name,MODEL_NAME)))
        cand_ids = [unhash_did(item) for item in cand_ids]
        print("文件读取完毕--------：  ",task_name)
        return cand_features, cand_ids
    except Exception as e:
        print(str(e))
        return None, None

def get_query_info(task_name):
    query_features = torch.load(query_feature_path_format.format(MODEL_NAME,task_name,MODEL_NAME))
    query_names = json.load(open(query_names_path_format.format(MODEL_NAME,task_name,MODEL_NAME)))
    return query_features, query_names

visualnews_task0_cand_features, visualnews_task0_cand_ids = get_cand_info("visualnews_task0")
mscoco_task0_cand_features, mscoco_task0_cand_ids = get_cand_info("mscoco_task0")
fashion200k_task0_cand_features, fashion200k_task0_cand_ids = get_cand_info("fashion200k_task0")
webqa_task1_cand_features, webqa_task1_cand_ids = get_cand_info("webqa_task1")
edis_task2_cand_features, edis_task2_cand_ids = get_cand_info("edis_task2")
webqa_task2_cand_features, webqa_task2_cand_ids = get_cand_info("webqa_task2")
visualnews_task3_cand_features, visualnews_task3_cand_ids = get_cand_info("visualnews_task3")
mscoco_task3_cand_features, mscoco_task3_cand_ids = get_cand_info("mscoco_task3")
fashion200k_task3_cand_features, fashion200k_task3_cand_ids = get_cand_info("fashion200k_task3")
nights_task4_cand_features, nights_task4_cand_ids = get_cand_info("nights_task4")
oven_task6_cand_features, oven_task6_cand_ids = get_cand_info("oven_task6")
infoseek_task6_cand_features, infoseek_task6_cand_ids = get_cand_info("infoseek_task6")
fashioniq_task7_cand_features, fashioniq_task7_cand_ids = get_cand_info("fashioniq_task7")
cirr_task7_cand_features, cirr_task7_cand_ids = get_cand_info("cirr_task7")
oven_task8_cand_features, oven_task8_cand_ids = get_cand_info("oven_task8")
infoseek_task8_cand_features, infoseek_task8_cand_ids = get_cand_info("infoseek_task8")

all_cand_ids = []
all_cand_ids.extend(visualnews_task0_cand_ids)
all_cand_ids.extend(mscoco_task0_cand_ids)
all_cand_ids.extend(fashion200k_task0_cand_ids)
all_cand_ids.extend(webqa_task1_cand_ids)
all_cand_ids.extend(edis_task2_cand_ids)
all_cand_ids.extend(webqa_task2_cand_ids)
all_cand_ids.extend(visualnews_task3_cand_ids)
all_cand_ids.extend(mscoco_task3_cand_ids)
all_cand_ids.extend(fashion200k_task3_cand_ids)
all_cand_ids.extend(nights_task4_cand_ids)
all_cand_ids.extend(oven_task6_cand_ids)
all_cand_ids.extend(infoseek_task6_cand_ids)
all_cand_ids.extend(fashioniq_task7_cand_ids)
all_cand_ids.extend(cirr_task7_cand_ids)
all_cand_ids.extend(oven_task8_cand_ids)
all_cand_ids.extend(infoseek_task8_cand_ids)

all_cand_features = []
all_cand_features.append(visualnews_task0_cand_features)
all_cand_features.append(mscoco_task0_cand_features)
all_cand_features.append(fashion200k_task0_cand_features)
all_cand_features.append(webqa_task1_cand_features)
all_cand_features.append(edis_task2_cand_features)
all_cand_features.append(webqa_task2_cand_features)
all_cand_features.append(visualnews_task3_cand_features)
all_cand_features.append(mscoco_task3_cand_features)
all_cand_features.append(fashion200k_task3_cand_features)
all_cand_features.append(nights_task4_cand_features)
all_cand_features.append(oven_task6_cand_features)
all_cand_features.append(infoseek_task6_cand_features)
all_cand_features.append(fashioniq_task7_cand_features)
all_cand_features.append(cirr_task7_cand_features)
all_cand_features.append(oven_task8_cand_features)
all_cand_features.append(infoseek_task8_cand_features)
all_cand_features = torch.cat(all_cand_features, dim=0)

task_names = [
    'visualnews_task0', 'mscoco_task0', 'fashion200k_task0', 'webqa_task1', 'edis_task2', 
    'webqa_task2', 'visualnews_task3', 'mscoco_task3', 'fashion200k_task3', 'nights_task4', 
    'oven_task6', 'infoseek_task6', 'fashioniq_task7', 'cirr_task7', 'oven_task8', 'infoseek_task8']

print(task_names)
for task_name in task_names:
    qrels_path = f"/group/40077/chaxjli/Retrieve/LamRA/data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
    print("开始处理索引--------：  ",qrels_path)
    qrel, qid_to_taskid = load_qrel(qrels_path)
    query_features, query_names = get_query_info(task_name)

    k_lists = [1, 5, 10, 50]
    res = {}
    index = []
    scores = []

    query_features_np = query_features.cpu().numpy()
    all_cand_features_np = all_cand_features.cpu().numpy()
    # debug --------------------- start
    # 将特征转换为 float16 以节省显存（需确认模型输出是否支持）可选，对精度会有影响
    # all_cand_features_np = all_cand_features.cpu().numpy().astype('float16')
    # 配置多 GPU 参数
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True          # 关键！分片存储到多个GPU
    co.useFloat16 = False     # 使用半精度可选
    co.reserveVecs = all_cand_features_np.shape[0]  # 预分配显存
    
    # 创建 CPU 索引
    dim = all_cand_features_np.shape[1] 
    cpu_index = faiss.IndexFlatL2(dim)
    
    # 转换到所有 GPU（自动检测 8 个 GPU）
    gpu_index_flat = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
    
    # 添加数据（自动分片到 8 个 GPU）
    gpu_index_flat.add(all_cand_features_np)
    # debug ----------------------- end

    # res = faiss.StandardGpuResources()  
    # dim = all_cand_features_np.shape[1]  
    # index_flat = faiss.IndexFlatL2(dim)
    # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)         # 作者代码
    # gpu_index_flat.add(all_cand_features_np)


    k = 50  
    scores = []
    indices = []

    batch_size = 1024  
    for i in tqdm(range(0, len(query_features_np), batch_size)):
        batch_query = query_features_np[i:i+batch_size]
        distances, topk_indexes = gpu_index_flat.search(batch_query, k)  
        scores.append(distances)
        indices.append(topk_indexes)

    scores = np.vstack(scores)    # (num_query, k)
    indices = np.vstack(indices)  # (num_query, k)

    cand_names = np.array([[all_cand_ids[item] for item in row] for row in indices])

    
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)

    with open(f"{save_dir_name}/{task_name}_scores.json", 'w') as f:
        json.dump(scores.tolist(), f, indent=2)

    with open(f"{save_dir_name}/{task_name}_query_names.json", 'w') as f:
        json.dump(query_names, f, indent=2)

    with open(f"{save_dir_name}/{task_name}_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)

    recall_res = {}
    for k in k_lists:
        recall_res[f'recall_{k}'] = []

    for ind, query_name in enumerate(tqdm(query_names)):
        relevant_docs = qrel[query_name]
        retrieved_indices_for_qid = cand_names[ind]
        for k in k_lists:
            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
            recall_res[f'recall_{k}'].append(recall_at_k)

    for k in k_lists:
        print(f"recall_at_{k} = {sum(recall_res[f'recall_{k}']) / len(recall_res[f'recall_{k}'])}")

    with open(f"{save_dir_name}/{MODEL_NAME}_mbeir_eval_global_results.txt", 'a') as f:
        f.write(qrels_path + '\n')
        for k in k_lists:
            f.write(f"recall_at_{k} = {sum(recall_res[f'recall_{k}']) / len(recall_res[f'recall_{k}'])}" + '\n')



