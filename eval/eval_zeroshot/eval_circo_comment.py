import os 
from transformers import AutoProcessor
import sys 
# 获取当前文件的绝对路径
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 计算上级目录的路径，用于导入自定义模块
module_path = os.path.join(current_file_path, "../../")
# 将上级目录添加到系统路径中，以便后续导入自定义模块
sys.path.append(module_path)
# 从自定义模块中导入 Qwen2VLRetForConditionalGeneration 类
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
import torch 
import argparse
# 从自定义模块中导入 CIRCODataset 类
from dataset.datasets_circo import CIRCODataset
# 从自定义模块中导入 EvalDataCollator 类
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
# 导入 Accelerator 类，用于分布式训练和推理
from accelerate import Accelerator
import accelerate
import numpy as np 
import json 
# 导入自定义的工具函数
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

# 定义评估函数，接收命令行参数作为输入
def eval(args):
    # 获取原始模型的 ID
    original_model_id = args.original_model_id
    # 获取要评估的模型的 ID
    model_id = args.model_id 
    # 从预训练模型中加载 Qwen2VLRetForConditionalGeneration 模型
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )

    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer
    model.use_instruction_mask = args.use_instruction_mask == 'True'
    model.use_bi_atten = args.use_bi_atten == 'True'
    model.mean_pooling = args.mean_pooling == 'True'
    model._set_model_use_bi_atten()

    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)       # 向分词器中添加新的标记，并返回添加的标记数量
        assert len(emb_tokens) == num_new_tokens                # 确保添加的标记数量与预期一致
        model.resize_token_embeddings(len(tokenizer))           # 调整模型的词嵌入层大小，以适应新添加的标记
        token_id = tokenizer.convert_tokens_to_ids(emb_token)   # 获取新添加标记的 ID
        if emb_token == "<instruction_start>":                  # 将其 ID 存储在模型配置中
            model.config.instruction_start_token_id = token_id
        elif emb_token == "<instruction_end>":
            model.config.instruction_end_token_id = token_id
        else:
            model.config.emb_token_id = token_id
    add_embed_token(tokenizer, model)
    if model.use_instruction_mask: 
        add_embed_token(tokenizer, model, emb_token="<instruction_start>")
        add_embed_token(tokenizer, model, emb_token="<instruction_end>")

    # 创建查询数据集对象
    query_dataset = CIRCODataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type='query',
        split=args.split
    )

    # 创建候选数据集对象
    cand_dataset = CIRCODataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type='image',
        split=args.split 
    )
    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor) # 创建查询数据的数据整理器
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)  # 创建候选数据的数据整理器
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    # 初始化 Accelerator 对象，用于分布式训练和推理
    accelerator = Accelerator(mixed_precision='bf16')
    # 获取当前设备
    device = accelerator.device 
    is_main_process = accelerator.is_main_process
    model.eval()  # 将模型设置为评估模式
    rank0_print("model.model.use_bi_atten: ", model.model.use_bi_atten)

    # 定义一个函数，用于将数据移动到指定设备上
    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)  # 如果是像素值，将其移动到指定设备并转换为指定数据类型
                else:
                    data[key] = data[key].to(device)            # 否则，仅将其移动到指定设备
        return data 


    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    from tqdm import tqdm 
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
        for batch in tqdm(query_dataloader, disable=not is_main_process): 
            batch = tensors_to_device(batch, device)                      # 将批次数据移动到指定设备上
            query_embed, batch_query_ids = model(**batch, inference=True) # 对批次数据进行推理，获取查询嵌入和查询 ID
            query_embed = F.normalize(query_embed, dim=-1)                # 对查询嵌入进行归一化处理
            query_embed = accelerator.gather_for_metrics(query_embed)     # 使用 Accelerator 收集查询嵌入
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)] # 使用 Accelerator 收集查询 ID
            query_ids.extend(batch_query_ids)                                                    # 将查询 ID 添加到列表中
            query_features.append(query_embed)                                                   # 将查询嵌入添加到列表中
            
        model.use_instruction_mask = False  
        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)  # 对批次数据进行推理，获取候选嵌入和候选 ID
            candidate_embed = F.normalize(candidate_embed, dim=-1) # 对候选嵌入进行归一化处理
            candidate_embed = accelerator.gather_for_metrics(candidate_embed) # 使用 Accelerator 收集候选嵌入
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)] # 使用 Accelerator 收集候选 ID
            candidate_ids.extend(batch_candidate_ids)  # 将候选 ID 添加到列表中
            candidate_features.append(candidate_embed) # 将候选嵌入添加到列表中

    
    query_features = torch.cat(query_features, dim=0)           # 将查询特征列表拼接成一个张量
    candidate_features = torch.cat(candidate_features, dim=0)   # 将候选特征列表拼接成一个张量

    if is_main_process:
        
        query_ids = np.array(query_ids)                                     # 将查询 ID 转换为 numpy 数组
        sorted_query_indices = np.argsort(query_ids)                        # 获取查询 ID 的排序索引
        query_features = query_features[sorted_query_indices]               # 根据排序索引对查询特征进行排序
        candidate_ids = np.array(candidate_ids)                             # 将候选 ID 转换为 numpy 数组
        sorted_candidate_indices = np.argsort(candidate_ids)                # 获取候选 ID 的排序索引
        candidate_features = candidate_features[sorted_candidate_indices]   # 根据排序索引对候选特征进行排序
        ap_at5, ap_at10, ap_at25, ap_at50 = [], [], [], []                              # 初始化不同召回率下的平均精度列表
        precision_at5, precision_at10, precision_at25, precision_at50 = [], [], [], []  # 初始化不同召回率下的准确率列表
        recall_at5, recall_at10, recall_at25, recall_at50 = [], [], [], []              # 初始化不同召回率下的召回率列表

        
        annotations = query_dataset.annotations                      # 获取查询数据集的注释信息
        max_num_gts = query_dataset.max_num_gts                      # 获取查询数据集的最大真实标签数量
        index_names = [str(item) for item in query_dataset.img_ids]  # 获取查询数据集的图像 ID 列表
        assert len(annotations) == len(query_features)               # 确保注释信息的数量与查询特征的数量一致
        save_dir_name = args.save_dir_name
        print("save_dir_name: ", save_dir_name)
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name, exist_ok=True)
        model_name = args.model_id.split('/')[-1]

        if args.split == 'val':
            for index in range(len(query_features)):
                # 获取目标图像的 ID
                target_img_id = str(annotations[index]['target_img_id'])
                # 获取真实图像的 ID 列表
                gt_img_ids = [str(x) for x in annotations[index]['gt_img_ids']]
                # 填充真实图像的 ID 列表，使其长度达到最大真实标签数量
                gt_img_ids += [''] * (max_num_gts - len(gt_img_ids))
                # 过滤掉空的真实图像 ID
                gt_img_ids = np.array(gt_img_ids)[np.array(gt_img_ids) != '']
                # 计算查询特征与候选特征的相似度得分
                score = query_features[index] @ candidate_features.T 
                # 获取得分最高的前 50 个候选图像的索引
                sorted_indices = torch.topk(score, dim=-1, k=50).indices.cpu()
                # 根据索引获取对应的图像名称
                sorted_index_names = np.array(index_names)[sorted_indices]
                # 判断每个候选图像是否为真实图像
                map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
                # 计算准确率
                precisions = torch.cumsum(map_labels, dim=0) * map_labels
                # 计算平均准确率
                precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)
                # 计算不同召回率下的平均精度
                ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
                ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
                ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
                ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

                # 确保目标图像 ID 在真实图像 ID 列表中
                assert target_img_id == gt_img_ids[0], f"Target name not in GTs {target_img_id} {gt_img_ids}"
                # 判断每个候选图像是否为目标图像
                single_gt_labels = torch.tensor(sorted_index_names == target_img_id)
                # 计算不同召回率下的召回率
                recall_at5.append(float(torch.sum(single_gt_labels[:5])))
                recall_at10.append(float(torch.sum(single_gt_labels[:10])))
                recall_at25.append(float(torch.sum(single_gt_labels[:25])))
                recall_at50.append(float(torch.sum(single_gt_labels[:50])))
            # 计算不同召回率下的平均精度均值
            map_at5 = np.mean(ap_at5) * 100
            map_at10 = np.mean(ap_at10) * 100
            map_at25 = np.mean(ap_at25) * 100
            map_at50 = np.mean(ap_at50) * 100
            # 计算不同召回率下的召回率均值
            recall_at5 = np.mean(recall_at5) * 100
            recall_at10 = np.mean(recall_at10) * 100
            recall_at25 = np.mean(recall_at25) * 100
            recall_at50 = np.mean(recall_at50) * 100
            # 打印不同召回率下的平均精度均值
            print('map_at5: ', map_at5)
            print('map_at10: ', map_at10)
            print('map_at25: ', map_at25)
            print('map_at50: ', map_at50)
            # 创建保存结果的目录
            save_dir_name = args.save_dir_name
            if not os.path.exists(save_dir_name):
                os.makedirs(save_dir_name, exist_ok=True)
            # 获取模型名称
            model_name = args.model_id.split('/')[-1]
            # 将评估结果写入文件
            with open(f"{save_dir_name}/{model_name}.txt", 'w') as f:
                f.write('circo evaluation' + '\n')
                f.write(f"map_at5: {map_at5}" + '\n')
                f.write(f"map_at10: {map_at10}" + '\n')
                f.write(f"map_at25: {map_at25}" + '\n')
                f.write(f"map_at50: {map_at50}" + '\n')

        elif args.split == 'test':
            # 初始化结果字典
            res = {}
            # 遍历每个查询特征
            for index in range(len(query_features)):
                # 计算查询特征与候选特征的相似度得分
                score = query_features[index] @ candidate_features.T 
                # 获取得分最高的前 50 个候选图像的索引
                sorted_indices = torch.topk(score, dim=-1, k=50).indices.cpu()
                # 根据索引获取对应的图像名称
                sorted_index_names = np.array(index_names)[sorted_indices]
                # 将图像名称列表转换为列表
                sorted_index_names = sorted_index_names.tolist()
                # 将查询 ID 和对应的候选图像名称列表存储在结果字典中
                res[annotations[index]['id']] = sorted_index_names

            
            # 将结果字典保存为 JSON 文件
            with open(f"{save_dir_name}/circo_test_retrieval_results.json", 'w') as f:
                json.dump(res, f)
            # 如果需要保存用于重排序的数据
            if args.save_for_rerank:
                save_for_rerank(query_features, candidate_features, query_ids, index_names, save_dir_name)

# 定义保存用于重排序数据的函数
def save_for_rerank(query_features, candidate_features, query_ids, index_names, save_dir_name):
    # 初始化索引列表
    index = []
    # 初始化得分列表
    scores = []
    # 遍历每个查询特征
    for i in range(len(query_features)):
        # 计算查询特征与候选特征的相似度得分
        score = query_features[i] @ candidate_features.T 
        # 获取得分最高的前 100 个候选图像的得分和索引
        topk_score, topk_indexes = torch.topk(score, k=100, dim=-1)
        # 将索引转换为列表
        topk_indexes = topk_indexes.squeeze().tolist()
        # 将索引添加到索引列表中
        index.append(topk_indexes)
        # 将得分添加到得分列表中
        scores.append(topk_score.tolist())
    
    # 根据索引获取对应的图像名称
    cand_names = np.array([[index_names[item] for item in row] for row in index])
    # 获取查询名称
    query_names = query_ids 

    # 将查询名称保存为 JSON 文件
    with open(f"{save_dir_name}/circo_test_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    # 将候选图像名称保存为 JSON 文件
    with open(f"{save_dir_name}/circo_test_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    # 将得分保存为 JSON 文件
    with open(f"{save_dir_name}/circo_test_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加注释文件路径前缀参数
    parser.add_argument('--annotation_path_prefix', type=str)
    # 添加图像文件路径前缀参数
    parser.add_argument('--image_path_prefix', type=str)
    # 添加原始模型 ID 参数
    parser.add_argument('--original_model_id', type=str)
    # 添加要评估的模型 ID 参数
    parser.add_argument('--model_id', type=str)
    # 添加数据集划分参数
    parser.add_argument('--split', type=str)
    # 添加批次大小参数
    parser.add_argument('--batch_size', type=int)
    # 添加是否保存用于重排序数据的参数
    parser.add_argument('--save_for_rerank', action='store_true')
    # 添加保存结果的目录名称参数
    parser.add_argument('--save_dir_name', type=str, help='Directory name to save results')
    # 添加是否使用指令掩码的参数
    parser.add_argument('--use_instruction_mask', type=str, default='True', help='Whether to use instruction mask')
    # 添加是否使用双向注意力的参数
    parser.add_argument('--use_bi_atten', type=str, default='True', help='Whether to use bi attention')
    # 添加是否使用均值池化的参数
    parser.add_argument('--mean_pooling', type=str, default='True', help='Whether to use mean pooling')

    # 解析命令行参数
    args = parser.parse_args()
    eval(args)