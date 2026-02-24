import json
from transformers import AutoProcessor
from collections import OrderedDict
import sys 
import os 
import numpy as np
import pickle 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration 
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
import torch
# 不确定是否需要适配
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
import argparse
from dataset.datasets_mmeb import MMEBEvalDataset,_load_data_jsonl
from collators.mmeb_eval import MMEBEvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
from tqdm import tqdm
# 导入自定义的工具函数 debug --------------------------------------------------------------------------------
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import time
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
# ------------------------------------------------------------------------------------------------
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

# 为每个 token 创建独立配置项 --------------------------------------------------------------------------------
def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    assert len(emb_tokens) == num_new_tokens
    model.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(emb_token)
    if emb_token == "<instruction_start>":
        model.config.instruction_start_token_id = token_id
    elif emb_token == "<instruction_end>":
        model.config.instruction_end_token_id = token_id
    else:
        model.config.emb_token_id = token_id  # 默认通用 token
# ------------------------------------------------------------------------------------------------
def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )
    # 处理模型的配置项--------------------------------------------------------------------------------
    model.mean_pooling = args.model_mean_pooling == "True"
    model.use_bi_atten = args.model_use_bi_atten == "True"
    model.use_latent_atten = args.model_use_latent_atten == "True"
    model.use_instruction_mask = args.model_use_instruction_mask == "True"
    
    
    # 加载 latent_atten 模块--------------------------------------------------------------------------
    latent_atten_path = os.path.join(model_id, "latent_atten.bin")
    if os.path.isfile(latent_atten_path):
        assert model.use_latent_atten == True, "模型配置文件中 use_latent_atten 为 False，但是路径当中存在 latent_atten 模型"
        latent_atten_state_dict = OrderedDict()
        ori_ckpts = torch.load(latent_atten_path)
        for k, v in ori_ckpts.items():
            # 这里的 base_model.model. 是因为在保存的时候，模型的 key 值是这样的
            latent_atten_state_dict[k.replace("base_model.model.", "")] = v
        model.load_state_dict(latent_atten_state_dict, strict=False)
    # ------------------------------------------------------------------------------------------------

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer 
    tokenizer.model_max_length = args.model_max_length
    add_embed_token(tokenizer, model)
    if model.use_instruction_mask: # 这里暂时先这样处理，后续再优化
        add_embed_token(tokenizer, model, emb_token="<instruction_start>")
        add_embed_token(tokenizer, model, emb_token="<instruction_end>")
    # ------------------------------------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process
    rank0_print("模型初始化完成————————————————————————————————————————————————————————————————————————")
    rank0_print("mean_pooling: ",model.mean_pooling)
    rank0_print("use_bi_atten: ",model.use_bi_atten)
    rank0_print("use_instruction_mask: ",model.use_instruction_mask)
    rank0_print("模型的类名是：",model.__class__.__name__)
    rank0_print("脚本开始的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rank0_print("脚本的参数是: ",args)
    # ------------------------------------------------------------------------------------------------
    for name, param in model.named_parameters():
        if "temp" in name:
            rank0_print("温度参数:", name)
            rank0_print("温度参数的值:", param)
    # ------------------------------------------------------------------------------------------------
    for idx, subset in enumerate(subset_names):
        rank0_print("*"*100)
        if subset in Classification:
            rank0_print(f"当前 {subset} 是 Classification 任务------------------------------------")
        elif subset in VQA:
            rank0_print(f"当前 {subset} 是 VQA 任务------------------------------------")
        elif subset in Retrieval:
            rank0_print(f"当前 {subset} 是 Retrieval 任务------------------------------------")
        elif subset in Visual_Grounding:
            rank0_print(f"当前 {subset} 是 Visual_Grounding 任务------------------------------------")
        else:
            rank0_print(f"当前 {subset} 是未知任务，请检查任务分类------------------------------------")
        encode_output_path = os.path.join(args.save_dir_name, "mmeb")
        score_path = os.path.join(encode_output_path, f"{subset}_score.json")
        # # 检查是否已有评估结果，避免重复计算
        # if os.path.exists(score_path):
        #     try:
        #         with open(score_path, "r") as f:
        #             score_dict = json.load(f)
        #         rank0_print(f"Found previous eval score, skipping {subset}")
        #         rank0_print(score_dict)
        #         continue
        #     except Exception as e:
        #         rank0_print(f"Failed to load existing score: {e}")
        #         pass
        rank0_print(f"{idx+1}/{len(subset_names)}: Processing {subset} now!")
        # 检查是否已有编码结果
        encode_qry_path = os.path.join(encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            rank0_print(f"Found existing encoded tensors, skipping encoding for {subset}")
            continue
        # 准备查询(query)和目标(target)数据集
        query_dataset = MMEBEvalDataset(
            subset=subset,
            text_field="qry_text",  # 查询文本字段
            img_path_field="qry_img_path",  # 查询图像路径字段
            tokenizer=tokenizer,
            has_instruction=True,
            use_instruction_token=True,
        )
        cand_dataset = MMEBEvalDataset(
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
            tokenizer=tokenizer,
            has_instruction=True if subset in Retrieval or subset in Visual_Grounding else False,
            use_instruction_token=True if subset in Retrieval or subset in Visual_Grounding else False,
        )
        query_data_collator = MMEBEvalDataCollator(tokenizer=tokenizer, processor=processor, \
                                                has_instruction=query_dataset.has_instruction, \
                                                use_instruction_token=query_dataset.use_instruction_token)
        cand_data_collator = MMEBEvalDataCollator(tokenizer=tokenizer, processor=processor,
                                                has_instruction=cand_dataset.has_instruction, \
                                                use_instruction_token=cand_dataset.use_instruction_token)
    
        query_dataloader = DataLoader(query_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=query_data_collator)
        candidate_dataloader = DataLoader(cand_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=cand_data_collator)
        model.eval()
        def tensors_to_device(data, device, dtype=model.dtype):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if key == 'pixel_values':
                        data[key] = data[key].to(device).to(dtype)
                    else:
                        data[key] = data[key].to(device)
            return data
        query_features = []
        query_ids = []
        candidate_features = []
        candidate_ids = []
        with torch.no_grad():
            candidate_batch_times = 0 # 收集候选集前 20 个batch 的时间
            query_batch_times = 0     # 收集查询集前 20 个batch 的时间
            # 将数据加载器和模型移到加速器设备上, 此处自动将数据集进行分布式处理
            query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model) 
            rank0_print("query_dataloader 和 candidate_dataloader 已经准备好，模型也已经准备好")
            # 处理查询集数据，此处必须将 use_instruction_mask 设置为 还原为原来的值
            model.use_instruction_mask = (args. model_use_instruction_mask == "True")
            rank0_print("查询集 model.use_instruction_mask: ",model.use_instruction_mask)
            for batch_idx,batch in enumerate(tqdm(query_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                query_embed, batch_query_ids = model(**batch, inference=True)
                query_embed = F.normalize(query_embed, dim=-1)
                query_embed = accelerator.gather_for_metrics(query_embed)
                batch_query_ids = accelerator.gather_for_metrics(batch_query_ids)[:len(query_embed)]
                # batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
                query_ids.extend(batch_query_ids)
                query_features.append(query_embed.cpu())  # 替换原append语句
                if batch_idx == 19:
                    query_batch_times = time.time() - start_time
                    rank0_print(f"查询集前 20 个batch的时间: {query_batch_times/60.00} min")
                    query_batch_times = query_batch_times * len(query_dataloader) / 20
                    rank0_print(f"查询集所有 batch的时间: {query_batch_times/3600.00} h")
            query_features = torch.cat(query_features, dim=0).cpu().detach().float().numpy()
            rank0_print("查询集的特征已经提取完成")
            rank0_print("查询集的特征形状: ", query_features.shape, "查询集的 ID 数量: ", len(set(query_ids)))
            query_paired_data = {item["id"]: (item["text"], item["image"]) for item in query_dataset.paired_data} 
            query_text_img_ids = [query_paired_data[id] for id in query_ids]
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((query_features, query_text_img_ids), f)  # 保存编码结果和对应数据
            rank0_print("查询集的编码结果已经保存到: ", encode_qry_path)
            
            # 处理候选池数据，此处必须将 use_instruction_mask 设置为 False
            if subset in Retrieval or subset in Visual_Grounding:
                model.use_instruction_mask = (args.model_use_instruction_mask == "True")
            else:
                model.use_instruction_mask = False
            rank0_print("候选集 model.use_instruction_mask: ",model.use_instruction_mask)
            for batch_idx,batch in enumerate(tqdm(candidate_dataloader, disable=not is_main_process)):
                if batch_idx == 0:
                    start_time = time.time()
                batch = tensors_to_device(batch, device)
                candidate_embed,batch_candidate_ids = model(**batch, inference=True)
                candidate_embed = F.normalize(candidate_embed, dim=-1)
                candidate_embed = accelerator.gather_for_metrics(candidate_embed)
                batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
                # batch_candidate_ids = accelerate.utils.gather_object(batch_candidate_ids)[:len(candidate_embed)]
                candidate_ids.extend(batch_candidate_ids)
                candidate_features.append(candidate_embed.cpu())  # 替换原append语句
                if batch_idx == 19:
                    candidate_batch_times = time.time() - start_time
                    rank0_print(f"候选集前 20 个 batch 的时间: {candidate_batch_times/60.00} min")
                    candidate_batch_times = candidate_batch_times * len(candidate_dataloader) / 20
                    rank0_print(f"候选集所有 batch 的时间: {candidate_batch_times/3600.00} h")
            candidate_features = torch.cat(candidate_features, dim=0).cpu().detach().float().numpy()
            rank0_print("候选集的特征已经提取完成")
            rank0_print("候选集的特征形状: ", candidate_features.shape, "候选集的 ID 数量: ", len(candidate_ids))
            candidate_paired_data = {item["id"]: (item["text"], item["image"]) for item in cand_dataset.paired_data}
            candidate_text_img_ids = [candidate_paired_data[id] for id in candidate_ids]
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((candidate_features, candidate_text_img_ids), f)  # 保存编码结果和对应数据
            rank0_print("候选集的编码结果已经保存到: ", encode_tgt_path)
    model = model.to("cpu")  # 先将模型移出 NPU
    del model
    accelerator.free_memory()  # 关键！释放加速器持有的资源
    torch.npu.empty_cache()
    if is_main_process:        
        # 计算评估分数
        for subset in tqdm(subset_names, desc="calculate score"):
            # 加载编码结果
            encode_qry_path = os.path.join(encode_output_path, f"{subset}_qry")
            encode_tgt_path = os.path.join(encode_output_path, f"{subset}_tgt")
            with open(encode_qry_path, 'rb') as f:
                qry_tensor, qry_index = pickle.load(f)  # 查询编码张量和索引
            with open(encode_tgt_path, 'rb') as f:
                tgt_tensor, tgt_index = pickle.load(f)  # 目标编码张量和索引
            # 构建查询和目标的映射字典
            qry_dict, tgt_dict = {}, {}
            for qry_t, tt in zip(qry_tensor, qry_index):
                text, img_path = tt[0], tt[1]
                qry_dict[(text, img_path)] = qry_t  # 键: (文本, 图像路径), 值: 编码向量
            rank0_print(f"qry_dict 的长度: {len(qry_dict)}")
            rank0_print(f"qry_dict 的示例: {list(qry_dict.items())[:1]}")
            for tgt_t, tt in zip(tgt_tensor, tgt_index):
                text, img_path = tt[0], tt[1]
                tgt_dict[(text, img_path)] = tgt_t
            rank0_print(f"tgt_dict 的长度: {len(tgt_index)}")
            rank0_print(f"tgt_dict 的示例: {list(tgt_dict.items())[:1]}")
            # 加载原始评估数据
            eval_data = _load_data_jsonl(os.path.join("/group/40077/Retrieval_Dataset/MMEB-eval",subset + "/" + subset + "_test.jsonl"))
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
                # scores, pred = get_pred(qry_t, tgt_t, normalization=False)  # 使用点积计算相似度
                scores, pred = get_pred(qry_t, tgt_t, normalization=True)
                # 判断预测是否正确（假设第一个候选为正确答案）
                if pred == 0:
                    n_correct += 1
                all_pred.append(all_candidates[pred])  # 保存预测结果
            # 保存预测结果
            with open(os.path.join(encode_output_path, f"{subset}_pred.txt"), "w") as f:
                for item in all_pred:
                    f.write(f"{item}\n")
            # 计算并保存评估分数
            score_path = os.path.join(encode_output_path, f"{subset}_score.json")
            rank0_print(f"Outputting final score to: {score_path}")
            with open(score_path, "w") as f:
                score_dict = {
                    "acc": n_correct/len(eval_data),  # 准确率
                    "num_correct": n_correct,         # 正确预测数
                    "num_pred": len(eval_data)        # 总预测数
                }
                json.dump(score_dict, f, indent=4)  # 保存为JSON格式
            rank0_print(f"{subset} accuracy: {n_correct/len(eval_data)}")  # 打印准确率
            rank0_print(f"{subset} num_correct: {n_correct}, num_pred: {len(eval_data)}")  # 打印正确预测数和总预测数
            rank0_print(f"Scores saved to: {score_path}")
            rank0_print("*"*100)
            result[subset] = n_correct/len(eval_data)
    # ------------------------------------------------------------------------------------------------
    # 计算 Classification、VQA、Retrieval、Visual Grounding、IND、OOD、Overall 的平均准确率
    Classification_acc = np.mean([result[task] for task in Classification if task in result])
    VQA_acc = np.mean([result[task] for task in VQA if task in result])
    Retrieval_acc = np.mean([result[task] for task in Retrieval if task in result])
    Visual_Grounding_acc = np.mean([result[task] for task in Visual_Grounding if task in result])
    IND_acc = np.mean([result[task] for task in IND if task in result])
    OOD_acc = np.mean([result[task] for task in OOD if task in result])
    Overall_acc = np.mean([result[task] for task in Overall if task in result])
    # 保存平均准确率
    rank0_print("开始计算各个任务的平均准确率")
    result["Classification"] = Classification_acc
    result["VQA"] = VQA_acc
    result["Retrieval"] = Retrieval_acc
    result["Visual_Grounding"] = Visual_Grounding_acc
    result["IND"] = IND_acc
    result["OOD"] = OOD_acc
    result["Overall"] = Overall_acc
    # V1-Overall = (10 * I-CLS + 10 * I-QA + 12 * I-RET + 4 * I-VG) / 36
    result["V1-Overall"] = (10 * Classification_acc + 10 * VQA_acc + 12 * Retrieval_acc + 4 * Visual_Grounding_acc) / 36

    rank0_print("")
    rank0_print(f"Classification accuracy: {Classification_acc}")
    rank0_print(f"VQA accuracy: {VQA_acc}")
    rank0_print(f"Retrieval accuracy: {Retrieval_acc}")
    rank0_print(f"Visual Grounding accuracy: {Visual_Grounding_acc}")
    rank0_print(f"IND accuracy: {IND_acc}")
    rank0_print(f"OOD accuracy: {OOD_acc}")
    rank0_print(f"Overall accuracy: {Overall_acc}")
    rank0_print("各个任务的平均准确率计算完成，且平均准确率已保存到结果字典中")
    # ------------------------------------------------------------------------------------------------    
    # 保存所有结果
    result_path = os.path.join(encode_output_path, "mmeb_eval_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    rank0_print(f"所有评估结果已保存到: {result_path}")
    rank0_print("脚本结束的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_dir_name', type=str)
    parser.add_argument('--query_dataset_has_instruction', type=str)
    parser.add_argument('--query_dataset_use_instruction_token', type=str)
    parser.add_argument('--model_mean_pooling', type=str)
    parser.add_argument('--model_use_bi_atten', type=str)
    parser.add_argument('--model_use_latent_atten', type=str)
    parser.add_argument('--model_use_instruction_mask', type=str)
    args = parser.parse_args()
    eval(args)



