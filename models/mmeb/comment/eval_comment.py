import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name

def batch_to_device(batch, device):
    """将批次数据转移到指定设备"""
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)  # 张量数据移至目标设备
        else:
            _batch[key] = value  # 非张量数据保持原样
    return _batch
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



def main():
    # 处理分布式训练参数格式，确保与Hugging Face解析器兼容
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    # 解析命令行参数为数据类实例
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    # 创建输出目录（用于存储编码结果和评估分数）
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # 加载模型处理器（用于图像和文本的预处理）
    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,  # 允许加载自定义模型代码
        num_crops=model_args.num_crops,  # 图像裁剪数量（用于多视图编码）
    )

    # 获取模型骨干类型（如LLaVA_NEXT、Phi3V等）
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')  # 打印骨干类型（仅主进程）

    # 加载预训练模型并配置为评估模式
    model = MMEBModel.load(model_args)
    model.eval()  # 设置为评估模式（关闭Dropout等训练特定层）
    model = model.to(training_args.device, dtype=torch.bfloat16)  # 模型移至指定设备并使用bf16精度

    # 初始化评估数据收集器（处理批次数据的组装）
    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    # 遍历所有数据集子集进行评估
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        
        # 检查是否已有评估结果，避免重复计算
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
                continue
            except Exception as e:
                print(f"Failed to load existing score: {e}")
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        
        # 检查是否已有编码结果
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            print(f"Found existing encoded tensors, skipping encoding for {subset}")
            continue

        # 准备查询(query)和目标(target)数据集
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",  # 查询文本字段
            img_path_field="qry_img_path",  # 查询图像路径字段
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",  # 目标文本字段
            img_path_field="tgt_img_path",  # 目标图像路径字段
        )

        # 创建数据加载器
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,  # 评估批次大小
            collate_fn=eval_collator,  # 数据收集函数
            shuffle=False,  # 不打乱数据顺序
            drop_last=False,  # 不丢弃最后不完整的批次
            num_workers=training_args.dataloader_num_workers,  # 数据加载工作线程数
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        # 编码查询数据
        encoded_tensor = []
        with torch.no_grad():  # 禁用梯度计算（节省内存，加速推理）
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                batch = batch_to_device(batch, training_args.device)  # 数据移至设备
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=batch)  # 模型前向传播（仅编码查询）
                encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())  # 保存编码结果
        encoded_tensor = np.concatenate(encoded_tensor)  # 合并所有批次的编码结果
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)  # 保存编码结果和对应数据

        # 编码目标数据（逻辑与查询编码相同）
        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch = batch_to_device(batch, training_args.device)
                output = model(tgt=batch)  # 模型前向传播（仅编码目标）
                encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    # 计算评估分数
    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        # 加载编码结果
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)  # 查询编码张量和索引
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)  # 目标编码张量和索引
        
        # 构建查询和目标的映射字典
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t  # 键: (文本, 图像路径), 值: 编码向量
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        # 加载原始评估数据
        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,  # 数据集分割（如"validation"）
        )
        
        # 计算准确率
        n_correct = 0  # 正确预测数
        all_pred = []  # 所有预测结果
        
        for row in eval_data:
            # 获取当前查询样本的编码向量
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            
            # 获取所有候选目标的编码向量
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            
            # 计算相似度并获取预测结果
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            
            # 判断预测是否正确（假设第一个候选为正确答案）
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])  # 保存预测结果
        
        # 保存预测结果
        with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        
        # 计算并保存评估分数
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {
                "acc": n_correct/len(eval_data),  # 准确率
                "num_correct": n_correct,  # 正确预测数
                "num_pred": len(eval_data)  # 总预测数
            }
            json.dump(score_dict, f, indent=4)  # 保存为JSON格式
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")  # 打印准确率


if __name__ == "__main__":
    main()