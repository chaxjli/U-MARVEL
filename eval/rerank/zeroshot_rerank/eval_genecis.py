import json 
import argparse 
from tqdm import tqdm 
import os
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)

args = parser.parse_args()
task_name = args.task_name 
model_name="qwen3-vl-4b_m-beir_stage1_model-Rank-Only-Pointwise"
source_model_name="qwen3-vl-4b_m-beir_stage3_model"
retrieval_path_name_fix=f"./result/{source_model_name}_eval_results_finetune/zeroshot/cc{task_name}"
rerank_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/{task_name}"
save_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/merge_retrieval_rerank_results/{task_name}_results.txt"

raw_scores = json.load(open(f"{retrieval_path_name_fix}/{task_name}_scores.json"))
rerank_scores = json.load(open(f"{rerank_path_name_fix}/{task_name}_top10_all_test_queryid2rerank_score.json"))

query_names = json.load(open(f"{retrieval_path_name_fix}/{task_name}_query_names.json"))
cand_names = json.load(open(f"{retrieval_path_name_fix}/{task_name}_cand_names.json"))


weight = 0.1
if task_name in ["genecis_change_object","genecis_focus_object"]:
    weight = 1.0
rerank_candidate_names = []
for idx, query_name in enumerate(query_names):
    raw_candidate_names = cand_names[idx][:10]
    raw_score = raw_scores[idx][:10]
    rerank_score = rerank_scores[str(query_name)]
    final_score = [1 * raw_score[index] + weight * rerank_score[index] for index in range(len(raw_score))]
    sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
    rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
    rerank_candidate_names.append(rerank_candidate_name)

k_lists = [1, 2, 3]
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

for ind, query_name in enumerate(tqdm(query_names)):
    relevant_docs = [0]
    retrieved_indices_for_qid = rerank_candidate_names[ind]
    for k in k_lists:
        recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
        res[f'recall_{k}'].append(recall_at_k)

for k in k_lists:
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")


# 保存结果
os.makedirs(os.path.dirname(save_path_name_fix), exist_ok=True)
with open(save_path_name_fix, 'w') as f:
    f.write(f"genecis {task_name[8:]} evaluation\n")
    for k in k_lists:
        f.write(f"recall_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}\n")

