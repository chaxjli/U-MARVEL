import numpy as np  # 导入NumPy库，用于数值计算
import os           # 导入操作系统库，用于文件路径操作
import json         # 导入JSON库，用于数据序列化和保存


def get_pred(qry_t, tgt_t, normalization=False):
    """
    计算查询向量与目标向量的相似度并获取预测结果
    
    参数:
        qry_t (np.ndarray): 查询向量（形状为 [dim,]）
        tgt_t (np.ndarray): 目标向量集合（形状为 [num_candidates, dim]）
        normalization (bool): 是否对向量进行归一化处理（默认: False）
    
    返回:
        scores (np.ndarray): 相似度得分数组（形状为 [num_candidates,]）
        pred (int): 预测的最佳匹配索引（得分最高的目标向量索引）
    
    数学原理:
        - 若 normalization=True:
          相似度计算为余弦相似度，公式为:
          score = (tgt_t · qry_t) / (||tgt_t|| × ||qry_t||)
          其中 ||·|| 表示L2范数，点积结果除以两个向量的范数乘积，结果范围为 [-1, 1]。
        
        - 若 normalization=False:
          相似度计算为点积，公式为:
          score = tgt_t · qry_t
          结果范围取决于向量的模长，可能为任意实数。
    """
    if normalization:
        # 计算查询向量的L2范数
        qry_t_norm = np.linalg.norm(qry_t)
        # 计算所有目标向量的L2范数（沿维度1，即每个样本的范数）
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        # 计算点积并除以范数乘积（余弦相似度）
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        # 直接计算点积作为相似度得分
        scores = np.dot(tgt_t, qry_t)
    # 获取得分最高的目标向量索引（预测结果）
    pred = np.argmax(scores)
    return scores, pred


def save_results(results, model_args, data_args, train_args):
    """
    将评估结果保存为JSON文件
    
    参数:
        results (dict): 评估结果字典，键为数据集名称，值为准确率（如 {"CIFAR-10": 0.85}）
        model_args (ModelArguments): 模型配置参数
        data_args (DataArguments): 数据配置参数
        train_args (TrainingArguments): 训练配置参数
    
    文件命名规则:
        文件名格式为:
        {model_name}_{model_type}_{embedding_type}_results.json
        - model_name: 模型名称（如 "llava-next-13b"）
        - model_type: 模型类型（可选，若未指定则为空字符串）
        - embedding_type: 嵌入类型（如 "text-image"）
    
    保存路径:
        结果保存到 data_args.encode_output_path 指定的目录中。
    """
    # 构建文件名
    save_file = (
        f"{model_args.model_name}"
        f"_{model_args.model_type if model_args.model_type is not None else ''}"
        f"_{data_args.embedding_type}_results.json"
    )
    # 创建保存路径（若不存在）
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    # 写入JSON文件
    with open(os.path.join(data_args.encode_output_path, save_file), "w") as json_file:
        json.dump(results, json_file, indent=4)  # indent=4 使JSON格式更易读


def print_results(results):
    """
    在控制台打印评估结果
    
    参数:
        results (dict): 评估结果字典，键为数据集名称，值为准确率
    """
    print("数据集,准确率")  # 打印表头
    for dataset, acc in results.items():
        # 格式化输出：数据集名称和准确率（保留4位小数）
        print(f"{dataset},{acc:.4f}")