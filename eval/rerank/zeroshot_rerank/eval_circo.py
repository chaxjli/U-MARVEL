import json 
import os 
import torch 
import numpy as np 
from tqdm import tqdm
task_name = "circo"
model_name="qwen3-vl-4b_m-beir_stage1_model-Rank-Only-Pointwise"
source_model_name="qwen3-vl-4b_m-beir_stage3_model"
retrieval_path_name_fix=f"./result/{source_model_name}_eval_results_finetune/zeroshot/{task_name}"
rerank_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/{task_name}"
save_path_name_fix=f"./result/result_rank/{model_name}/{source_model_name}/zeroshot/merge_retrieval_rerank_results/{task_name}_results.txt"


raw_scores = json.load(open(f"{retrieval_path_name_fix}/{task_name}_test_scores_val.json"))
# rerank_scores = json.load(open(f"{rerank_path_name_fix}/{task_name}_top50_all_test_queryid2rerank_score_val.json"))
rerank_scores = json.load(open(f"{rerank_path_name_fix}/{task_name}_top50_all_test_queryid2rerank_score.json"))


query_names = json.load(open(f"{retrieval_path_name_fix}/{task_name}_test_query_names_val.json"))
cand_names = json.load(open(f"{retrieval_path_name_fix}/{task_name}_test_cand_names_val.json"))





rerank_candidate_names = []
all_sorted_indices = []
for idx, query_name in enumerate(query_names):
    raw_candidate_names = cand_names[idx][:50]
    raw_score = raw_scores[idx][:50]
    rerank_score = rerank_scores[str(query_name)]
    final_score = [1 * raw_score[index] + 0.1* rerank_score[index] for index in range(len(raw_score))]
    sorted_indices = [index for index, value in sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)]
    rerank_candidate_name = [raw_candidate_names[index] for index in sorted_indices]
    # rerank_candidate_name.extend(cand_names[idx][10:50])
    rerank_candidate_names.append(rerank_candidate_name)
    all_sorted_indices.append(sorted_indices)
    

ap_at5, ap_at10, ap_at25, ap_at50 = [], [], [], []
precision_at5, precision_at10, precision_at25, precision_at50 = [], [], [], []
recall_at5, recall_at10, recall_at25, recall_at50 = [], [], [], []
annotation_path_prefix = "./data/circo/annotations"
image_path_prefix = "./data/circo/images/unlabeled2017"
img_info_path = f"{annotation_path_prefix}/image_info_unlabeled2017.json"
with open(img_info_path) as f:
    imgs_info = json.load(f)
img_paths = [f"{image_path_prefix}/{img_info['file_name']}" for img_info in imgs_info['images']]
img_ids = [img_info["id"] for img_info in imgs_info["images"]]
img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}
annotation_file = f"{annotation_path_prefix}/val.json"
with open(annotation_file) as f:
    annotations = json.load(f)        
max_num_gts = 23
index_names = [str(item) for item in img_ids]

for index in tqdm(range(len(query_names))):
    target_img_id = str(annotations[index]['target_img_id'])
    gt_img_ids = [str(x) for x in annotations[index]['gt_img_ids']]
    gt_img_ids += [''] * (max_num_gts - len(gt_img_ids))
    gt_img_ids = np.array(gt_img_ids)[np.array(gt_img_ids) != '']
    # score = query_features[index] @ candidate_features.T
    # sorted_indices = torch.topk(score, dim=-1, k=10).indices.cpu()
    
    rerank_candidate_name = rerank_candidate_names[index]
    
    # sorted_indices = all_sorted_indices[index]
    # sorted_index_names = np.array(index_names)[sorted_indices]

    # print('sorted_index_names: ', sorted_index_names,"类型: ", type(sorted_index_names))
    # print('gt_img_ids: ', gt_img_ids, "类型: ", type(gt_img_ids))
    # print('target_img_id: ', target_img_id, "类型: ", type(target_img_id))
    # print("rerank_candidate_name: ", rerank_candidate_name, "类型: ", type(rerank_candidate_name))
    # if index == 10:
    #     break
    map_labels = torch.tensor(np.isin(rerank_candidate_name, gt_img_ids), dtype=torch.uint8)
    precisions = torch.cumsum(map_labels, dim=0) * map_labels

    precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)
    ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
    ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
    ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
    ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

    assert target_img_id == gt_img_ids[0], f"Target name not in GTs {target_img_id} {gt_img_ids}"

map_at5 = np.mean(ap_at5) * 100
map_at10 = np.mean(ap_at10) * 100
map_at25 = np.mean(ap_at25) * 100
map_at50 = np.mean(ap_at50) * 100

print('map_at5: ', map_at5)
print('map_at10: ', map_at10)
print('map_at25: ', map_at25)
print('map_at50: ', map_at50)

# 保存结果
os.makedirs(os.path.dirname(save_path_name_fix), exist_ok=True)
with open(save_path_name_fix, 'w') as f:
    f.write(f"{task_name} evaluation\n")
    f.write(f"map_at5 = {map_at5}\n")
    f.write(f"map_at10 = {map_at10}\n")
    f.write(f"map_at25 = {map_at25}\n")
    f.write(f"map_at50 = {map_at50}\n")

res = {}
for query_id, candidate_names in zip(query_names, rerank_candidate_names):
    res[query_id] = candidate_names

with open(f'{rerank_path_name_fix}/circo_test_rerank_results_val.json', 'w') as f:
    json.dump(res, f)