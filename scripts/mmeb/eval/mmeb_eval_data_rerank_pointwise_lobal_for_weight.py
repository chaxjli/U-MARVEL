import json 
import argparse 
from tqdm import tqdm 
import os
import copy
from collections import defaultdict
import collections
model_name = "qwen2-vl-7b_umarvel_hard_negative_mining_mmeb"                               # 这里需要修改: 需要第一阶段模型名称
rerank_model_name = "qwen2-vl-7b_umarvel_progressive_transition_mmeb-Rank-Only-Pointwise"  # 这里需要修改: 需要第二阶段模型名称
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
task_names = Overall[:]
task_name2metric = {task_name:"recall_1" for task_name in task_names}
k_lists = [1]
rerank_path_prefix = f"./result/result_rank/mmeb/{rerank_model_name}/{model_name}/mmeb_cand_without_inst/"
if not os.path.exists(rerank_path_prefix):
    os.makedirs(rerank_path_prefix)

# 从 0.0 到 m 之间每隔 0.1 取一个值，总共 (10m+1) 个值
# weight_params = [i*0.1 for i in range(0,31)]
weight_params = [0.1]
best_weight_param = 0.0
best_metric = 0
task_name2best_res = {} # key: task_name, value: best res
mmeb_result = {}

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
    for task_name in task_names:
        # 注意这里需要修改: 需要第一阶段模型评估时生成的文件，以及第二阶段 rerank 模型评估时生成的文件
        raw_scores = load_json(f"./result/mmeb/{model_name}_eval_results_finetune/mmeb_cand_without_inst/{task_name}_{model_name}_scores.json")
        rerank_scores = load_json(f"./result/result_rank/mmeb/{rerank_model_name}/{model_name}/mmeb_cand_without_inst/{task_name}_top50_test_queryid2rerank_score.json")
        query_names = load_json(f"./result/mmeb/{model_name}_eval_results_finetune/mmeb_cand_without_inst/{task_name}_{model_name}_query_names.json")
        cand_names = load_json(f"./result/mmeb/{model_name}_eval_results_finetune/mmeb_cand_without_inst/{task_name}_{model_name}_cand_names.json")
        qrels_path = f"./data/MMEB-eval/{task_name}/{task_name}_qrel.json"
        qrel= load_qrel(qrels_path)
        rerank_candidate_names = []
        for idx, query_name in enumerate(query_names):
            rerank_num = min(50, len(rerank_scores[query_name]))
            raw_candidate_names = cand_names[idx][:rerank_num]  # 取前50个候选
            raw_score = raw_scores[idx][:rerank_num]         # 取前50个候选的分数
            rerank_score = rerank_scores[query_name]
            final_score = [1.0* raw_score[index] + weight_param * rerank_score[index] for index in range(len(raw_score))]
            sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
            rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
            rerank_candidate_names.append(rerank_candidate_name)
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
    print("打印文件保存的位置：", os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'))
    if avg_metric > best_metric:
        best_metric = avg_metric
        best_weight_param = weight_param
        task_name2best_res = copy.deepcopy(task_name2res)
        for task_name, res in task_name2best_res.items():
            mmeb_result[task_name] = sum(res[f'recall_{1}']) / len(res[f'recall_{1}'])
# 打印并保存最佳结果
print(f"best_weight_param: {best_weight_param}")
print(f"best_metric: {best_metric}")
print("打印文件保存的位置：", os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'))
with open(os.path.join(rerank_path_prefix,'weight_pointwise_recall_results.txt'), 'a') as file:  # 改为追加模式
        file.write(f"best_weight_param: {best_weight_param}, best_metric: {best_metric}\n")
# 保存最佳结果到文件
for task_name, res in task_name2best_res.items():
    save_results_to_file_add(task_name, k_lists, res)
    print(f"task_name: {task_name}")
    for k in k_lists:
        print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")    

mmeb_result = {task_name: float(value) for task_name, value in mmeb_result.items()}
mmeb_result["Classification"] = sum(mmeb_result[task_name] for task_name in Classification) / len(Classification)
mmeb_result["VQA"] = sum(mmeb_result[task_name] for task_name in VQA) / len(VQA)
mmeb_result["Retrieval"] = sum(mmeb_result[task_name] for task_name in Retrieval) / len(Retrieval)
mmeb_result["Visual_Grounding"] = sum(mmeb_result[task_name] for task_name in Visual_Grounding) / len(Visual_Grounding)
mmeb_result["IND"] = sum(mmeb_result[task_name] for task_name in IND) / len(IND)
mmeb_result["OOD"] = sum(mmeb_result[task_name] for task_name in OOD) / len(OOD)
mmeb_result["Overall"] = sum(mmeb_result[task_name] for task_name in Overall) / len(Overall)    
mmeb_result["V1-Overall"] = (10 * mmeb_result["Classification"] + 10 * mmeb_result["VQA"] + 12 * mmeb_result["Retrieval"] + 4 * mmeb_result["Visual_Grounding"]) / 36
with open(os.path.join(rerank_path_prefix,'mmeb_eval_results.json'), 'w') as file:  # 改为写入模式
    json.dump(mmeb_result, file, indent=4)


    
    

