import json 
import argparse 
from tqdm import tqdm 
import os

model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr" # 这里需要修改: 需要第一阶段模型名称
rerank_model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr-Rank" # 这里需要修改: 需要第二阶段 rerank 模型名称

task_names = [
    'visualnews_task0', 'mscoco_task0', 'fashion200k_task0', 'webqa_task1', 'edis_task2', 
    'webqa_task2', 'visualnews_task3', 'mscoco_task3', 'fashion200k_task3', 'nights_task4', 
    'oven_task6', 'infoseek_task6', 'fashioniq_task7', 'cirr_task7', 'oven_task8', 'infoseek_task8']

task_name2metric = {
    'mscoco_task0': "recall_5",
    'mscoco_task3': "recall_5",
    'visualnews_task0': "recall_5",
    'fashion200k_task0': "recall_10",
    'webqa_task1': "recall_5",
    'edis_task2': "recall_5",
    'webqa_task2': "recall_5",
    'visualnews_task3': "recall_5",
    'fashion200k_task3': "recall_10",
    'nights_task4': "recall_5",
    'infoseek_task6': "recall_5",
    'fashioniq_task7': "recall_10",
    'cirr_task7': "recall_5",
    'oven_task8': "recall_5",
    'infoseek_task8': "recall_5",
    'oven_task6': "recall_5"
}
# # 从 0.0 到 2.0 之间每隔 0.01 取一个值，总共 201 个值
# weight_params = [i*0.1 for i in range(0,200)]
# print(len(weight_params))


# print(len(weight_param_dict),len(task_name2metric))
# best_weight_param = weight_param_dict.get(task_name)
# metric = task_name2metric.get(task_name)
# best_metric = 0
# best_res = {}

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    relevant_docs_set = set(relevant_docs)
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

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

# 定义一个函数将权重和指标结果写入文件
def write_weight_and_results_to_file(weight_param, task_name, k_lists, res):
    with open(os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"weight_param: {weight_param}\n")
        file.write(f"{task_name}\n")
        for k in k_lists:
            recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
            line = f"recall_at_{k} = {recall_at_k}\n"
            file.write(line)

for task_name in task_names:        
    raw_scores = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_scores.json"))
    rerank_scores = json.load(open(f"./result/result_rank/{rerank_model_name}/{model_name}/local/{task_name}_top50_test_queryid2rerank_score.json"))
    query_names = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_query_names.json"))
    cand_names = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_cand_names.json"))
    qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
    qrel, _ = load_qrel(qrels_path)
    rerank_candidate_names = []
    if 'mscoco' not in task_name:
        weight_param = 1.0
    else:
        weight_param = 0.5 if 'mscoco_task0' in task_name else 0.1
    for idx, query_name in enumerate(query_names):
        raw_candidate_names = cand_names[idx][:50]
        raw_score = raw_scores[idx][0][:50]
        rerank_score = rerank_scores[query_name]
        # rerank_score = rerank_scores_debug[query_name]
        final_score = [1 * raw_score[index] + weight_param * rerank_score[index] for index in range(len(raw_score))]
        sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
        rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
        rerank_candidate_names.append(rerank_candidate_name)
    k_lists = [1, 5, 10, 20]
    res = {}

    for k in k_lists:
        res[f'recall_{k}'] = []

    for ind, query_name in enumerate(tqdm(query_names)):
        relevant_docs = qrel[query_name]
        retrieved_indices_for_qid = rerank_candidate_names[ind]
        for k in k_lists:
            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
            res[f'recall_{k}'].append(recall_at_k)

    for k in k_lists:
        print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

    rerank_path_prefix = f"./result/result_rank/{rerank_model_name}/{model_name}/local/"
    with open(os.path.join(rerank_path_prefix,'pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"{task_name}\n")
        for k in k_lists:
            recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
            line = f"recall_at_{k} = {recall_at_k}\n"
            file.write(line)

