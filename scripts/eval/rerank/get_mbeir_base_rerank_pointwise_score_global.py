import json 
import argparse 
from tqdm import tqdm 
import os
import copy

# 这里需要修改: 需要第一阶段模型名称
model_name = "qwen2-vl-7b_umarvel_hard_negative_mining" # 提供第一阶段检索结果的模型名称
rerank_model_name = "qwen2-vl-7b_umarvel_progressive_transition_m-beir-Rank-Only-Pointwise" # 这里需要修改: 需要第二阶段 rerank 模型名称

task_names = [
    'visualnews_task0', 
    'mscoco_task0', 'fashion200k_task0', 'webqa_task1', 'edis_task2', 
    'webqa_task2', 'visualnews_task3', 'mscoco_task3', 'fashion200k_task3', 'nights_task4', 
    'oven_task6',  'fashioniq_task7', 'cirr_task7', 'oven_task8', 'infoseek_task8',
    'infoseek_task6',
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

rerank_path_prefix = f"./result/score/{rerank_model_name}/{model_name}/global"
if not os.path.exists(rerank_path_prefix):
    os.makedirs(rerank_path_prefix)


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

# key: query_name, value: {did: cand_name, score: score}
from collections import defaultdict
fusion_results = defaultdict(dict)

for task_name in task_names:
    if 'mscoco' not in task_name:
        weight_param = 1.0
    else:
        weight_param = 0.5 if 'mscoco_task0' in task_name else 0.1
    # 注意这里需要修改: 需要第一阶段模型评估时生成的文件，以及第二阶段 rerank 模型评估时生成的文件
    raw_scores = load_json(f"./result/result_eval_train/{model_name}/global/{task_name}_scores.json")
    rerank_scores = load_json(f"./result/result_rank/train/{rerank_model_name}/{model_name}/global/{task_name}_top50_train_queryid2rerank_score.json")
    query_names = load_json(f"./result/result_eval_train/{model_name}/global/{task_name}_query_names.json")
    cand_names = load_json(f"./result/result_eval_train/{model_name}/global/{task_name}_cand_names.json")
    qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
    
    fusion_query_names = list()  # 收集融合后的 qid
    fusion_cand_names = list()   # 收集融合正确排序的 did
    fusion_scores = list()       # 收集融合后的分数

    for idx, query_name in enumerate(query_names):
        raw_candidate_names = cand_names[idx][:50]
        raw_score = raw_scores[idx][50:100][::-1]  # 取出前50个候选的分数
        # rerank_scores 是一个字典，key 是 qid，value 是一个列表: 对应着 cand_names 当中各个 qid 的分数
        rerank_score = rerank_scores[query_name]
        final_score = [1.0* raw_score[index] + weight_param * rerank_score[index] for index in range(len(raw_score))]
        sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
        fusion_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
        fusion_score = [final_score[index] for index in sorted_indices]
        
        fusion_query_names.append(query_name)
        fusion_cand_names.append(fusion_candidate_name)
        fusion_scores.append(fusion_score)
        fusion_results[query_name] = {"qid":query_name,"did": copy.deepcopy(fusion_candidate_name), "score": copy.deepcopy(fusion_score)}

    print(f"len(fusion_query_names): {len(fusion_query_names)}")
    print(f"len(fusion_cand_names): {len(fusion_cand_names)}")
    print(f"len(fusion_scores): {len(fusion_scores)}")

    # 保存融合后的结果
    assert len(fusion_query_names) == len(fusion_cand_names) == len(fusion_scores), "融合后的结果长度不一致"
    with open(f"{rerank_path_prefix}/{task_name}_fusion_scores.json", "w") as f:
        json.dump(fusion_scores, f,ensure_ascii=False)
    with open(f"{rerank_path_prefix}/{task_name}_fusion_query_names.json", "w") as f:
        json.dump(fusion_query_names, f, ensure_ascii=False)    
    with open(f"{rerank_path_prefix}/{task_name}_fusion_cand_names.json", "w") as f:
        json.dump(fusion_cand_names, f, ensure_ascii=False)
    print(f"融合后的结果已保存到 {rerank_path_prefix}/{task_name}_fusion_scores.json")
    print(f"融合后的结果已保存到 {rerank_path_prefix}/{task_name}_fusion_query_names.json")
    print(f"融合后的结果已保存到 {rerank_path_prefix}/{task_name}_fusion_cand_names.json")

fusion_results_list = []
data_sample_file_path = "/group/40077/Retrieval_Dataset/M-BEIR/query/union_train/mbeir_union_up_train_10percent.jsonl"
with open(data_sample_file_path, 'r', encoding='utf-8') as f:
    # 统计文件行数
    line_count = sum(1 for _ in f)
    f.seek(0)  # 将文件指针移回文件开头
    for index,line in enumerate(tqdm(f, total = line_count,desc=f"Loading: ")):
        item = json.loads(line)
        qid = item['qid']
        assert qid in fusion_results, f"qid {qid} 不在 fusion_results2 中"
        fusion_results_list.append(copy.deepcopy(fusion_results[qid]))

# 将 fusion_results_list 保存成 jsonl 文件
with open(f"{rerank_path_prefix}/fusion_results.jsonl", "w") as f:
    for item in fusion_results_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"融合后的结果已保存到 {rerank_path_prefix}/fusion_results.jsonl")









    
    

