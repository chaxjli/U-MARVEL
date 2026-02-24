import json 
import argparse 
from tqdm import tqdm 
import os
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--data_type', type=str, default=None)

args = parser.parse_args()
task_name = args.task_name 
data_type = args.data_type
model_name="qwen3-vl-4b_m-beir_stage1_model-Rank-Only-Pointwise"
source_model_name="qwen3-vl-4b_m-beir_stage3_model"
if data_type is not None:
    retrieval_path_name_fix=f"./result/{source_model_name}_eval_results_finetune/zeroshot/{task_name}_{data_type}"
    save_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/merge_retrieval_rerank_results/{task_name}_{data_type}_results.txt"
else:
    retrieval_path_name_fix=f"./result/{source_model_name}_eval_results_finetune/zeroshot/{task_name}"
    save_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/merge_retrieval_rerank_results/{task_name}_results.txt"

rerank_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/{task_name}"
# 保存结果
os.makedirs(os.path.dirname(save_path_name_fix), exist_ok=True)
with open(save_path_name_fix, 'w') as f:
    if data_type is not None:
        f.write(f"{task_name.split('_')[0]} {task_name.split('_')[1]} {data_type} evaluation\n")
    else:
        f.write(f"{task_name} evaluation\n")
    
    if args.task_name == 'ccneg':
        raw_score = json.load(open(f"{retrieval_path_name_fix}/{task_name}_scores.json"))
        rerank_score = json.load(open(f"{rerank_path_name_fix}/{task_name}_top2_all_test_queryid2rerank_score.json"))
        beat = 0
        for i in tqdm(range(len(rerank_score) // 2)):
            score1 = 1 * raw_score[2 * i] + 1 * rerank_score[2 * i]
            score2 = 1 * raw_score[2 * i + 1] + 1 * rerank_score[2 * i + 1]
            if score1 > score2:
                beat += 1
        print(beat / (len(rerank_score) // 2))
        f.write(f"accuracy = {beat / (len(rerank_score) // 2)}\n")
    
    else:
        raw_score = json.load(open(f"{retrieval_path_name_fix}/sugar_crepe_{data_type}.json"))
        rerank_score = json.load(open(f"{rerank_path_name_fix}/sugar_crepe_top2_all_{data_type}_test_queryid2rerank_score.json"))
        beat = 0
        for i in tqdm(range(len(rerank_score) // 2)):
            score1 = 1 * raw_score[2 * i] + 0.1 * rerank_score[2 * i]
            score2 = 1 * raw_score[2 * i + 1] + 0.1 * rerank_score[2 * i + 1]
            if score1 > score2:
                beat += 1
        print(beat / (len(rerank_score) // 2))
        f.write(f"accuracy = {beat / (len(rerank_score) // 2)}\n")