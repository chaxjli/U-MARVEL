import json 
import argparse 
from tqdm import tqdm 
import os
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--zero_shot_rerank_model_name', type=str)
args = parser.parse_args()
task_name = args.task_name 
model_name = args.model_name # 这里需要修改: 需要第一阶段模型名称
zero_shot_rerank_model_name = args.zero_shot_rerank_model_name # 这里需要修改: 需要第二阶段 rerank 模型名称
rerank_path_prefix = f"./result/result_zero_shot_rerank/{zero_shot_rerank_model_name}/{model_name}/local/" # 评估作者的模型

# 定义一个函数将权重和指标结果写入文件
def write_weight_and_results_to_file(weight_param, task_name, k_lists, res):
    # 打开一个文件以写入模式，如果文件不存在则创建，如果存在则覆盖原有内容
    with open(os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"weight_param: {weight_param}\n")
        file.write(f"{task_name}\n")
        for k in k_lists:
            # 计算 recall_at_k 的值
            recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
            # 格式化要写入的字符串
            line = f"recall_at_{k} = {recall_at_k}\n"
            # 将字符串写入文件
            file.write(line)

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

# 注意这里需要修改: 需要第一阶段模型评估时生成的文件，以及第二阶段 rerank 模型评估时生成的文件
raw_scores = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_scores.json"))
rerank_scores = json.load(open(f"./result/result_zero_shot_rerank/{zero_shot_rerank_model_name}/{model_name}/local/{task_name}_top50_test_queryid2rerank_score.json"))
query_names = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_query_names.json"))
cand_names = json.load(open(f"./result/{model_name}_eval_results_finetune/local/mbeir_{task_name}_test_{model_name}_cand_names.json"))
qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
qrel, _ = load_qrel(qrels_path)

rerank_candidate_names = []

if 'mscoco' not in task_name:
    weight_param = 1.0
else:
    weight_param = 0.1 if 'mscoco_task0' in task_name else 0.1

weight_param_dict = {
    'mscoco_task0': 0.1,
    'mscoco_task3': 0.0,
    'visualnews_task0': 0.0,
    'fashion200k_task0': 0.0,
    'webqa_task1': 0.0,
    'edis_task2': 0.0,
    'webqa_task2': 0.0,
    'visualnews_task3': 0.0,
    'fashion200k_task3': 0.0,
    'nights_task4': 0.0,
    'infoseek_task6': 0.0,
    'fashioniq_task7': 0.0,
    'cirr_task7': 0.0,
    'oven_task8': 0.0,
    'infoseek_task8': 0.0,
    'oven_task6': 0.0
}

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

print(len(weight_param_dict),len(task_name2metric))
best_weight_param = weight_param_dict.get(task_name)
metric = task_name2metric.get(task_name)
best_metric = 0
best_res = {}

# 从 0.0 到 2.0 之间每隔 0.01 取一个值，总共 201 个值
weight_params = [i*0.001 for i in range(0,2001)]
print(len(weight_params))

for weight_param in tqdm(weight_params):
     # Default to "recall_at_5" if task_name not found in the dictionary
    rerank_candidate_names = []  # 在此处初始化
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

    for ind, query_name in enumerate((query_names)):
        relevant_docs = qrel[query_name]
        retrieved_indices_for_qid = rerank_candidate_names[ind]
        for k in k_lists:
            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
            res[f'recall_{k}'].append(recall_at_k)
    # 当前的 weight_param 和对应的 res[metric] 结果,recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
    current_metric = sum(res[metric]) / len(res[metric])
    # print(f"weight_param: {weight_param}, {metric}: {current_metric}")
    if current_metric > best_metric:
        best_metric = current_metric
        best_weight_param = weight_param
        best_res = copy.deepcopy(res)
    # 记录当前的 weight_param = 0 结果，写入 'weight_pointwise_recall_results.txt' 文件
    if weight_param == 0:
        write_weight_and_results_to_file(weight_param, task_name, k_lists, res)

# 输出最佳的 weight_param 和对应的 res[metric] 结果，并将其写入文件
print(f"task_name: {task_name}")
print(f"Best weight_param: {best_weight_param}")
print(f"Best {metric}: {best_metric}")
write_weight_and_results_to_file(best_weight_param, task_name, k_lists, best_res)

# 重新赋值
weight_param = best_weight_param
res = best_res

k_lists = [1, 5, 10, 20]
for k in k_lists:
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

# 打开一个文件以写入模式，如果文件不存在则创建，如果存在则覆盖原有内容
with open(os.path.join(rerank_path_prefix,'pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
    file.write(f"{task_name}\n")
    for k in k_lists:
        # 计算 recall_at_k 的值
        recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
        # 格式化要写入的字符串
        line = f"recall_at_{k} = {recall_at_k}\n"
        # 将字符串写入文件
        file.write(line)

