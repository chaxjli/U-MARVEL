import os 
import json 
from tqdm import tqdm 
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor
import sys 
import argparse
import torch.nn.functional as F
import copy
def load_json(file_path):
    """
    加载 json 文件
    :param file_path: json 文件路径
    :return: json 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_jsonl(file_path):
    """
    加载 jsonl 文件
    :param file_path: jsonl 文件路径
    :return: jsonl 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def merge_json_files(args):
    """
    1. 检查 local_data.json/global_data.json 文件是否和 cand_names,query_names 文件对应；
    2. 将 data.json 文件和 scores.json 文件合并成 data_with_scores 文件；
    3. 将所有的 data_with_scores 文件合并成一个文件 jsonl 文件；
    4. 将所有的 data_with_scores 文件合并成一个文件 json 文件；
    5. 删除中间生成的文件 data_with_scores 文件；
    """
    # path_prefix: ./result/result_eval_train/held-out/10percent/qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr/local
    path_prefix = args.path_prefix
    files = os.listdir(path_prefix)
    bash_json_name = "local_data.json" if "local" in path_prefix else "global_data.json"
    files = [file for file in files if file.endswith(bash_json_name)]
    files = [os.path.join(path_prefix, file) for file in files]
    print("files 文件个数: ", len(files))
    cand_names_files = [file.replace(bash_json_name, 'cand_names.json') for file in files]
    query_names_files = [file.replace(bash_json_name, 'query_names.json') for file in files]
    scores_files = [file.replace(bash_json_name, 'scores.json') for file in files]
    data_with_scores_files = [file.replace(bash_json_name, 'data_with_scores.json') for file in files]
    success_files = []
    fail_files = []
    
    # 进行第一步，检查 data.json 文件是否和 cand_names,query_names 文件对应
    qid2top100_cands = {}
    qid2top100_scores = {}
    for data_file, cand_name_file, query_name_file in zip(files, cand_names_files, query_names_files):
        try:
            data = load_json(data_file)
            data_cand_names = load_json(cand_name_file)
            data_query_names = load_json(query_name_file)
            print(f"data_file: {data_file}, cand_name_file: {cand_name_file}, query_name_file: {query_name_file}")
            # 检查 data.json 文件是否和 cand_names,query_names 文件对应
            for index, item in tqdm(enumerate(data.items())):
                qid, value = item
                gt_docs = value["gt_docs"]
                top100_docs = value["top100_docs"]
                assert qid == data_query_names[index], f"数据不匹配: {qid} != {data_query_names[index]}"
                assert top100_docs == data_cand_names[index], f"数据不匹配: {top100_docs} != {data_cand_names[index]}"
                qid2top100_cands[qid] = data_cand_names[index][:] # 收集 qid 到 top100_cands 的映射关系
            success_files.append(data_file)
            print(f"文件 {data_file} 检查成功！")
        except Exception as e:
            print(f"文件 {data_file} 检查失败！")
            fail_files.append(data_file)
            continue
    print(f"成功检查的文件个数: {len(success_files)}")
    print(f"成功检查的文件: {success_files}")
    print(f"失败检查的文件个数: {len(fail_files)}")
    print(f"失败检查的文件: {fail_files}")
    
    # 进行第二步，将 data.json 文件和 scores.json 文件合并成 data_with_scores 文件
    for data_file, scores_file, data_with_scores_file in zip(files, scores_files, data_with_scores_files):
        data = load_json(data_file)
        scores = load_json(scores_file)
        data_with_scores = {}
        print(f"data_file: {data_file}, scores_file: {scores_file}, data_with_scores_file: {data_with_scores_file}")
        for index, item in tqdm(enumerate(data.items())):
            qid, value = item
            gt_docs = value["gt_docs"]
            top100_docs = value["top100_docs"]
            if "local" in path_prefix:
                score = scores[index][0]
            elif "global" in path_prefix:
                score = scores[index]
            else:
                raise ValueError("path_prefix 必须包含 local 或 global")            
            assert len(top100_docs) == len(score), f"数据不匹配: {len(top100_docs)} != {len(score)}"
            qid2top100_scores[qid] = score[:] # 收集 qid 到 top100_scores 的映射关系
            data_with_scores[qid] = {
                "qid": qid,
                "gt_docs": gt_docs,
                "top100_docs": top100_docs,
                "score": score
            }
            
        with open(data_with_scores_file, 'w') as f:
            json.dump(data_with_scores, f, ensure_ascii=False)
        print(f"文件 {data_with_scores_file} 生成成功！")
    
    # 进行第三步，将所有的 data_with_scores 文件合并成一个文件, 处理成列表形式
    # union_up_train_path = "/group/40077/Retrieval_Dataset/M-BEIR/query/union_train/mbeir_union_up_train.jsonl"
    union_up_train_path = "/group/40077/Retrieval_Dataset/M-BEIR/query/union_train/mbeir_union_up_train_10percent_held-out.jsonl"
    data_query = load_jsonl(union_up_train_path)
    rerank_all_data = {} # 收集 data_with_scores_files 文件当中的数据
    for file in tqdm(data_with_scores_files):
        data = load_json(file)
        for item in data: # 遍历字典
            rerank_all_data[item] = copy.deepcopy(data[item])
    print("rerank_all_data 个数应该是 66834, 实际是: ", len(rerank_all_data))
    data_query_with_top100_score = []
    for item in tqdm(data_query): # data_query 的数据是训练集 query
        qid = item["qid"]
        pos_list = item["pos_cand_list"]
        assert qid in rerank_all_data, f"数据不匹配: {qid} 不在 rerank_all_data 中"
        assert pos_list == rerank_all_data[qid]["gt_docs"], f"数据不匹配: {pos_list} != {rerank_all_data[qid]['gt_docs']}"
        # 对 data_query_with_top100_score 做一下校验
        assert qid2top100_cands[qid] == rerank_all_data[qid]["top100_docs"], f"数据不匹配: {qid2top100_cands[qid]} != {rerank_all_data[qid]['top100_docs']}"
        assert qid2top100_scores[qid] == rerank_all_data[qid]["score"], f"数据不匹配: {qid2top100_scores[qid]} != {rerank_all_data[qid]['score']}"
        data_query_with_top100_score.append(copy.deepcopy(rerank_all_data[qid]))
    
    # data_query_with_top100_score 保存为 jsonl 文件
    print("data_query_with_top100_score 个数应该是 69317, 实际是: ", len(data_query_with_top100_score))
    print("data_query_with_top100_score[0]: ", data_query_with_top100_score[0])
    if "local" in path_prefix:
        with open(path_prefix + '/' + 'rerank_data_all_eval_train_local.jsonl', 'w') as f:
            for item in data_query_with_top100_score:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("rerank_data_all_eval_train_local.jsonl 文件生成成功！")
    elif "global" in path_prefix:   
        with open(path_prefix + '/' + 'rerank_data_all_eval_train_global.jsonl', 'w') as f:
            for item in data_query_with_top100_score:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("rerank_data_all_eval_train_global.jsonl 文件生成成功！")
    else:
        raise ValueError("path_prefix 必须包含 local 或 global")
    
    # 第四步：将所有的 data_with_scores 文件合并成一个文件
    print("data_with_scores 文件个数: ", len(data_with_scores_files))
    all_data_with_scores = {} # 收集 data_with_scores_files 文件当中的数据
    for file in tqdm(data_with_scores_files):
        data = load_json(file)
        all_data_with_scores.update(data)
    # all_data_with_scores 遍历这个字典，删除每个元素的 score 这个键
    for item in all_data_with_scores.keys():
        all_data_with_scores[item].pop("score")
    print("all_data_with_scores 个数应该是 66834, 实际是: ", len(all_data_with_scores))
    if "local" in path_prefix:
        with open(path_prefix + '/' + 'rerank_data_all_eval_train_local.json', 'w') as f:
            json.dump(all_data_with_scores, f,ensure_ascii=False)
        print("rerank 训练数据 rerank_data_all_eval_train_local.json 文件生成成功！")
    elif "global" in path_prefix:
        with open(path_prefix + '/' + 'rerank_data_all_eval_train_global.json', 'w') as f:
            json.dump(all_data_with_scores, f,ensure_ascii=False)
        print("rerank 训练数据 rerank_data_all_eval_train_global.json 文件生成成功！")
    else:
        raise ValueError("path_prefix 必须包含 local 或 global")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_prefix', type=str, help='path prefix of the json files')
    args = parser.parse_args()
    merge_json_files(args)

    