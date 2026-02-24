# —————————— 导入必要的库 ——————————
import json  # 用于处理 JSON 数据
import argparse  # 用于解析命令行参数
from tqdm import tqdm  # 用于显示进度条
import os  # 用于处理文件和目录路径
import sys

# —————————— 解析命令行参数 ——————————
# 创建一个 ArgumentParser 对象，用于解析命令行参数
parser = argparse.ArgumentParser()
# 添加一个名为 --task_name 的参数，类型为字符串
parser.add_argument('--task_name', type=str)

# 解析命令行参数
args = parser.parse_args()
# 获取命令行中指定的任务名称
task_name = args.task_name

# —————————— 定义加载 qrel 文件的函数 ——————————
def load_qrel(filename):
    """
    加载 qrel 文件，该文件包含查询 ID、文档 ID、相关性得分和任务 ID 等信息。
    :param filename: qrel 文件的路径
    :return: 一个字典 qrel，键为查询 ID，值为相关文档 ID 的列表；一个字典 qid_to_taskid，键为查询 ID，值为任务 ID
    """
    qrel = {}  # 用于存储查询 ID 到相关文档 ID 列表的映射
    qid_to_taskid = {}  # 用于存储查询 ID 到任务 ID 的映射
    # 打开文件以只读模式读取
    with open(filename, "r") as f:
        # 逐行读取文件内容
        for line in f:
            # 按空格分割每行内容，得到查询 ID、占位符、文档 ID、相关性得分和任务 ID
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            # 假设只有正的相关性得分表示相关文档
            if int(relevance_score) > 0:
                # 如果查询 ID 不在 qrel 字典中，初始化一个空列表
                if query_id not in qrel:
                    qrel[query_id] = []
                # 将相关文档 ID 添加到该查询 ID 对应的列表中
                qrel[query_id].append(doc_id)
                # 如果查询 ID 不在 qid_to_taskid 字典中，记录该查询 ID 对应的任务 ID
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    # 打印从文件中加载的查询数量
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    # 打印每个查询的平均相关文档数量
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

# —————————— 加载 qrel 数据 ——————————
# 构建 qrel 文件的路径
qrels_path = f"./data/M-BEIR/qrels/test/mbeir_{task_name}_test_qrels.txt"
# 调用 load_qrel 函数加载 qrel 数据，只使用返回的 qrel 字典，忽略 qid_to_taskid
qrel, _ = load_qrel(qrels_path)

# —————————— 加载查询名称和候选名称数据 ——————————

# 加载查询名称的 JSON 文件
query_names = json.load(open(f"./LamRA_Ret_eval_results/mbeir_{task_name}_test_qwen2-vl-7b_LamRA-Ret_query_names.json"))

# 加载候选名称的 JSON 文件
cand_names = json.load(open(f"./LamRA_Ret_eval_results/mbeir_{task_name}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json"))

# —————————— 确定重排序数据的路径并加载 ——————————
# 评估作者的模型时的重排序文件路径前缀
rerank_path_prefix = "./result/mbeir_rerank_files" 

# 评估我们的模型时的重排序文件路径前缀（当前注释掉）
# rerank_path_prefix = "./result/mbeir_rerank_files" 

# 构建重排序数据的文件路径并加载
rerank_data = json.load(open(f"{rerank_path_prefix}/{task_name}_top5_all_test_queryid2rerank_outputs_listwise.json"))

# —————————— 根据重排序结果确定最终的候选名称 ——————————
rerank_candidate_names = []  # 用于存储重排序后的候选名称
# 遍历每个查询名称

# 加载query_names和cand_names后
assert len(query_names) == len(cand_names), f"query_names({len(query_names)})和cand_names({len(cand_names)})长度不一致"
print(f"query_names({len(query_names)})和cand_names({len(cand_names)})长度一致")

for idx, query_name in enumerate(query_names):
    # 获取该查询的前 5 个候选名称
    raw_candidate_names = cand_names[idx][:5]
    # 获取该查询的重排序结果
    top1_name = rerank_data[query_name]
    # 根据重排序结果选择对应的候选名称
    if "1" in top1_name:
        rerank_candidate_names.append([raw_candidate_names[0]])
    elif "2" in top1_name:
        rerank_candidate_names.append([raw_candidate_names[1]])
    elif "3" in top1_name:
        rerank_candidate_names.append([raw_candidate_names[2]])
    elif "4" in top1_name:
        rerank_candidate_names.append([raw_candidate_names[3]])
    else:
        rerank_candidate_names.append([raw_candidate_names[4]])
    # else:
    #     print(rerank_candidate_names[0])
    #     print((raw_candidate_names))
    #     print(len(rerank_data.keys()))
    #     print(query_name)
    #     print(top1_name)
    #     print(idx)
    #     break

print(f"rerank_candidate_names 的长度是：{len(rerank_candidate_names)}")
# sys.exit("Exit at this line") # debug 使用

# 如果不进行重排序，可以使用原始的候选名称（当前注释掉）
# rerank_candidate_names = cand_names 

# —————————— 定义要计算的召回率的 k 值列表 ——————————
k_lists = [1]  # 这里只计算 recall@1
res = {}  # 用于存储每个 k 值对应的召回率结果

# 初始化每个 k 值对应的召回率列表
for k in k_lists:
    res[f'recall_{k}'] = []

# —————————— 定义计算召回率的函数 ——————————
def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    """
    计算在给定 k 值下的召回率。
    :param relevant_docs: 相关文档 ID 的列表
    :param retrieved_indices: 检索到的文档 ID 的列表
    :param k: 要计算召回率的 k 值
    :return: 召回率，值为 0 或 1
    """
    # 如果没有相关文档，返回 0
    if not relevant_docs:
        return 0.0 
    # 获取前 k 个检索到的文档 ID 的集合
    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    # 将相关文档 ID 列表转换为集合
    relevant_docs_set = set(relevant_docs)
    # 检查相关文档集合和前 k 个检索到的文档集合是否有交集
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0  # 有交集则返回 1，表示成功检索到相关文档
    else:
        return 0.0  # 无交集则返回 0

# —————————— 计算每个查询的召回率 ——————————
# Traceback (most recent call last):
#   File "/group/40077/chaxjli/Retrieve/LamRA/eval/rerank/mbeir_rerank_listwise.py", line 82, in <module>
#     retrieved_indices_for_qid = rerank_candidate_names[ind]
# IndexError: list index out of range

# 遍历每个查询名称，使用 tqdm 显示进度条
assert len(query_names) == len(rerank_candidate_names), f"query_names({len(query_names)}) 长度不等于 rerank_candidate_names({len(rerank_candidate_names)})"
print(f"query_names 长度({len(query_names)})\n rerank_candidate_names长度：({len(rerank_candidate_names)})")



for ind, query_name in enumerate(tqdm(query_names)):
    # 获取该查询的相关文档 ID 列表
    relevant_docs = qrel[query_name]
    # 获取该查询的重排序后的候选名称列表
    retrieved_indices_for_qid = rerank_candidate_names[ind]
    # 遍历每个 k 值
    for k in k_lists:
        # 计算该查询在当前 k 值下的召回率
        recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
        # 将召回率添加到对应的 k 值的结果列表中
        res[f'recall_{k}'].append(recall_at_k)

# —————————— 打印每个 k 值的平均召回率 ——————————
for k in k_lists:
    # 计算该 k 值下的平均召回率
    print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

# —————————— 将结果写入文件 ——————————
# 打开文件以追加模式写入结果，如果文件不存在则创建
with open(os.path.join(rerank_path_prefix,'listwise_recall_results.txt'), 'a') as file:
    # 写入任务名称
    file.write(f"{task_name}\n")
    # 遍历每个 k 值
    for k in k_lists:
        # 计算该 k 值下的平均召回率
        recall_at_k = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
        # 格式化要写入的字符串
        line = f"recall_at_{k} = {recall_at_k}\n"
        # 将字符串写入文件
        file.write(line)




