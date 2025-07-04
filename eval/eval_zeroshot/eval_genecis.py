import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)

# 注意预训练模型也是使用这个数据集
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration # 作者的源代码,有问题吧，为什么拿这个模型评估
# 该使用 finetune 模型来进行评估吧,但是使用这个模型来评估效果会奇差，预训练也是如此
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration

import torch 
import argparse
from dataset.datasets_genecis import GenecisCOCODataset, GenecisVAWDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
import numpy as np 
import json 

# 导入自定义的工具函数 debug --------------------------------------------------------------------------------
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import time
# --------------------------------------------------------------------------------------------------------
def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )
    # 重置模型的参数
    model.use_instruction_mask = args.use_instruction_mask == 'True'
    model.use_bi_atten = args.use_bi_atten == 'True'
    model.mean_pooling = args.mean_pooling == 'True'
    model._set_model_use_bi_atten()

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer
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
    add_embed_token(tokenizer, model)
    if model.use_instruction_mask: # 这里暂时先这样处理，后续再优化
        add_embed_token(tokenizer, model, emb_token="<instruction_start>")
        add_embed_token(tokenizer, model, emb_token="<instruction_end>")
    # ------------------------------------------------------------------------------------------------


    if args.data_type == 'change_object':
        query_dataset = GenecisCOCODataset(type='query', annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
        cand_dataset = GenecisCOCODataset(type='image',  annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
    elif args.data_type == 'focus_object':
        query_dataset = GenecisCOCODataset(type='query', annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
        cand_dataset = GenecisCOCODataset(type='image',  annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
    elif args.data_type == 'change_attribute':
        query_dataset = GenecisVAWDataset(type='query',  annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
        cand_dataset = GenecisVAWDataset(type='image',   annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
    elif args.data_type == 'focus_attribute':
        query_dataset = GenecisVAWDataset(type='query',  annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)
        cand_dataset = GenecisVAWDataset(type='image',   annotation_path=args.annotation_path, image_path_prefix=args.image_path_prefix,tokenizer=tokenizer)

    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    model.eval()
    # debug --------------------------------------------------------------------------------------------
    rank0_print("模型初始化完成————————————————————————————————————————————————————————————————————————")
    # 打印 npu 的显存占用情况
    rank0_print(f"NPU 显存使用情况: {torch.npu.memory_allocated() / (1024 ** 2):.2f} MB")
    rank0_print("mean_pooling: ",model.mean_pooling ,"use_bi_atten: ",model.use_bi_atten, "use_latent_atten: ",model.use_latent_atten)
    rank0_print("model.model.use_bi_atten: ", model.model.use_bi_atten)
    rank0_print("模型的类名是：",model.__class__.__name__)
    rank0_print("-"*50)
    rank0_print("query_dataset 使用的咒语： ",query_dataset.prompt)
    rank0_print("cand_dataset 使用的咒语： ",cand_dataset.prompt)
    rank0_print("-"*50)
    rank0_print("脚本的运行时间是: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rank0_print("脚本的参数是: ",args)
    rank0_print("-"*50)
    # ------------------------------------------------------------------------------------------------

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

    from tqdm import tqdm 
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
        for batch in tqdm(query_dataloader, disable=not is_main_process):
            # model.use_instruction_mask = True  # 处理查询集数据，此处必须将 use_instruction_mask 设置为 True
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            # model.use_instruction_mask = False  # 处理候选集数据，此处必须将 use_instruction_mask 设置为 False
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)
    
    model = model.to("cpu")    # 先将模型移出 NPU
    del model
    accelerator.free_memory()  # 关键！释放加速器持有的资源
    torch.npu.empty_cache()    # 清空 NPU 缓存
    query_features = torch.cat(query_features, dim=0).to(device)
    candidate_features = torch.cat(candidate_features, dim=0).to(device)

    
    if is_main_process:
        # Adjust the order according to ids 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        query_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        candidate_features = candidate_features[sorted_candidate_indices]

        val_samples = query_dataset.val_samples
        gallery_ids = query_dataset.gallery_ids 
        topk = (1, 2, 3)

        query_ids = query_ids.tolist()

        res = {'recall_1': [], 'recall_2': [], 'recall_3': []}

        for i in range(len(query_features)):
            query_feature = query_features[i]
            gallery = val_samples[i]['gallery']
            if args.data_type == 'change_object' or args.data_type == 'focus_object':
                target_name = str(val_samples[i]['target']['val_image_id'])
                gallery_names = [str(item['val_image_id']) for item in gallery]
            elif args.data_type == 'change_attribute' or args.data_type == 'focus_attribute':
                target_name = f"{str(val_samples[i]['target']['image_id'])}_{i}_1.jpg"
                gallery_names = [f"{str(item['image_id'])}_{i}_{2 + ind}.jpg" for ind, item in enumerate(gallery)]
            gallery_and_target_names = [target_name]
            gallery_and_target_names.extend(gallery_names)
            gallery_and_target_indices = [gallery_ids.index(item) for item in gallery_and_target_names]
            gallery_and_target_features = torch.stack([candidate_features[ind] for ind in gallery_and_target_indices])

            score = query_feature @ gallery_and_target_features.T 
            _, sorted_idxs = score.sort(dim=-1, descending=True)

            for k in topk:
                if 0 in sorted_idxs[:k]:
                    res[f"recall_{k}"].append(1)
                else:
                    res[f"recall_{k}"].append(0)

        res['recall_1'] = sum(res['recall_1']) / len(res['recall_1'])
        res['recall_2'] = sum(res['recall_2']) / len(res['recall_2'])
        res['recall_3'] = sum(res['recall_3']) / len(res['recall_3'])

        print('recall_at1: ', res['recall_1'])
        print('recall_at2: ', res['recall_2'])  
        print('recall_at3: ', res['recall_3'])

        model_name = args.model_id.split('/')[-1]
        # debug 创建保存结果的目录 ------------------------------------------------------------
        save_dir_name = args.save_dir_name
        print("save_dir_name: ", save_dir_name)
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name, exist_ok=True)  # 递归创建目录，若存在则忽略
        # ----------------------------------------------------------------------------------
        with open(f"{save_dir_name}/{model_name}.txt", 'w') as f:
            f.write(f'genecis {args.data_type} evaluation' + '\n')
            for key in res:
                f.write(f"{key} = {res[key]}" + "\n")

        # save_for_rerank
        if args.save_for_rerank:
            save_for_rerank(query_features, candidate_features, val_samples, query_ids, gallery_ids, args.data_type, save_dir_name)


def save_for_rerank(query_features, candidate_features, val_samples, query_ids, gallery_ids, data_type, save_dir_name):
    cand_names = []
    scores = []
    for i in range(len(query_features)):
        query_feature = query_features[i]
        gallery = val_samples[i]['gallery']
        if data_type == 'change_object' or data_type == 'focus_object':
            target_name = str(val_samples[i]['target']['val_image_id'])
            gallery_names = [str(item['val_image_id']) for item in gallery]
        elif args.data_type == 'change_attribute' or args.data_type == 'focus_attribute':
            target_name = f"{str(val_samples[i]['target']['image_id'])}_{i}_1.jpg"
            gallery_names = [f"{str(item['image_id'])}_{i}_{2 + ind}.jpg" for ind, item in enumerate(gallery)]
        gallery_and_target_names = [target_name]
        gallery_and_target_names.extend(gallery_names)
        gallery_and_target_indices = [gallery_ids.index(item) for item in gallery_and_target_names]
        gallery_and_target_features = torch.stack([candidate_features[ind] for ind in gallery_and_target_indices])
        score = query_feature @ gallery_and_target_features.T 
        topk_score, topk_indexes = torch.topk(score, k=10, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        cand_names.append(topk_indexes)
        scores.append(topk_score.tolist())

    query_names = query_ids 
    with open(f"{save_dir_name}/genecis_{data_type}_query_names.json", 'w') as f:
        json.dump(query_names, f, indent=2)
    with open(f"{save_dir_name}/genecis_{data_type}_cand_names.json", 'w') as f:
        json.dump(cand_names, f, indent=2)
    with open(f"{save_dir_name}/genecis_{data_type}_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--save_for_rerank', action='store_true')
    parser.add_argument('--save_dir_name', type=str, help='Directory name to save results')
    parser.add_argument('--use_instruction_mask', type=str, default='True', help='Whether to use instruction mask')
    parser.add_argument('--use_bi_atten', type=str, default='True', help='Whether to use bi attention')
    parser.add_argument('--mean_pooling', type=str, default='True', help='Whether to use mean pooling')
    args = parser.parse_args()
    eval(args)