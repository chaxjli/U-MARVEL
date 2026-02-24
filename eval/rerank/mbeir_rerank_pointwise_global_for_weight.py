import json 
import argparse 
from tqdm import tqdm 
import os
import copy

# model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr"                                   # 这里需要修改: 需要第一阶段模型名称
# rerank_model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr-Rank"                     # 这里需要修改: 需要第二阶段 rerank 模型名称
# rerank_model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr-Rank-Only-Pointwise"        # 这里需要修改: 需要第二阶段 rerank 模型名称
model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature_continue_distillrerank-8m8g_1e-5"        # 这里需要修改: 需要第一阶段模型名称
rerank_model_name = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature_continue_distillrerank-8m8g_1e-5-Rank-Only-Pointwise"        # 这里需要修改: 需要第二阶段 rerank 模型名称
task_names = [
    'visualnews_task0', 
    'mscoco_task0', 'fashion200k_task0', 'webqa_task1', 'edis_task2', 
    'webqa_task2', 'visualnews_task3', 'mscoco_task3', 'fashion200k_task3', 'nights_task4', 
    'oven_task6', 'infoseek_task6', 'fashioniq_task7', 'cirr_task7', 'oven_task8', 'infoseek_task8'
    ]

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

k_lists = [1, 5, 10, 20]
rerank_path_prefix = f"./result/result_rank/{rerank_model_name}/{model_name}/global/"

# 从 0.0 到 m 之间每隔 0.1 取一个值，总共 (10m+1) 个值
# weight_params = [i*0.1 for i in range(0,21)]
weight_params = [0.1]
best_weight_param = 0.0
best_metric = 0
task_name2best_res = {} # key: task_name, value: best res

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0
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
    print(f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}")
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

# 定义一个函数将指标结果写入文件, 使用写入模式
def save_results_to_file_write(task_name, k_lists, res):
    with open(os.path.join(rerank_path_prefix,'pointwise_recall_results.txt'), 'r') as file:  # 改为追加模式
        file.write(f"{task_name}\n")
        for k in k_lists:
            recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
            line = f"recall_at_{k} = {recall_at_k}\n"
            file.write(line)

# 定义一个函数将指标结果写入文件, 使用追加模式
def save_results_to_file_add(task_name, k_lists, res):
    with open(os.path.join(rerank_path_prefix,'pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"{task_name}\n")
        for k in k_lists:
            recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
            line = f"recall_at_{k} = {recall_at_k}\n"
            file.write(line)

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

for weight_param in weight_params:
    task_name2res = {}      # key: task_name, value: res
    task_name2value = {}    # key: task_name, value: metric_value
    temp_weight_param = weight_param      
    for task_name in task_names:

        # if 'mscoco' not in task_name:
        #     weight_param = temp_weight_param
        # else:
        #     weight_param = 0.5 if 'mscoco_task0' in task_name else 0.1
        
        # if "mscoco_task0" in task_name:
        #     weight_param = 0.5
        # elif "mscoco_task3" in task_name:
        #     weight_param = 0.1
        # else:
        #     weight_param = temp_weight_param
        

        # 注意这里需要修改: 需要第一阶段模型评估时生成的文件，以及第二阶段 rerank 模型评估时生成的文件
        raw_scores = load_json(f"./result/{model_name}_eval_results_finetune/global/{task_name}_scores.json")
        rerank_scores = load_json(f"./result/result_rank/{rerank_model_name}/{model_name}/global/{task_name}_top50_test_queryid2rerank_score.json")
        query_names = load_json(f"./result/{model_name}_eval_results_finetune/global/{task_name}_query_names.json")
        cand_names = load_json(f"./result/{model_name}_eval_results_finetune/global/{task_name}_cand_names.json")
        qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
        qrel, _ = load_qrel(qrels_path)
        rerank_candidate_names = []
        for idx, query_name in enumerate(query_names):
            raw_candidate_names = cand_names[idx][:50]
            raw_score = raw_scores[idx][::-1]
            # print("len(raw_scores[idx]):", len(raw_scores[idx]))
            # print("raw_scores[idx]:", raw_scores[idx])
            rerank_score = rerank_scores[query_name]
            final_score = [1.0* raw_score[index] +  weight_param* rerank_score[index] for index in range(len(raw_score))]
            sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
            rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
            rerank_candidate_names.append(rerank_candidate_name)
            # print(f"query_name: {query_name}, raw_candidate_names: {raw_candidate_names}, rerank_candidate_names: {rerank_candidate_name}")
            # print(f"raw_score: {raw_score}, rerank_score: {rerank_score}, final_score: {final_score}")
            # print(f"sorted_indices: {sorted_indices}")
            # break
        res = {}
        for k in k_lists:
            res[f'recall_{k}'] = []
        for ind, query_name in enumerate(tqdm(query_names)):
            relevant_docs = qrel[query_name]
            retrieved_indices_for_qid = rerank_candidate_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)
        task_name2res[task_name] = copy.deepcopy(res)
        task_name2value[task_name] = sum(res[task_name2metric[task_name]]) / len(res[task_name2metric[task_name]])
        # write_weight_and_results_to_file(weight_param, task_name, k_lists, res)
        # 打印计算结果: 
        # for k in k_lists:
        #     print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")
    # 计算平均值 task_name2value 的平均值
    print(f"weight_param: {weight_param}")
    print(f"task_name2value: {task_name2value}")
    avg_metric = sum(task_name2value.values()) / len(task_name2value)
    print(f"average metric: {avg_metric}")
    with open(os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"weight_param: {weight_param}, average metric: {avg_metric}\n")
    if avg_metric > best_metric:
        best_metric = avg_metric
        best_weight_param = weight_param
        task_name2best_res = copy.deepcopy(task_name2res)

# 打印并保存最佳结果
print(f"best_weight_param: {best_weight_param}")
print(f"best_metric: {best_metric}")
with open(os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"best_weight_param: {best_weight_param}, best_metric: {best_metric}\n")

# 保存最佳结果到文件
for task_name, res in task_name2best_res.items():
    save_results_to_file_add(task_name, k_lists, res)
    print(f"task_name: {task_name}")
    for k in k_lists:
        print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")    





    
    

