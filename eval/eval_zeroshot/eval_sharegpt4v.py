import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)

from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration

import torch 
import argparse
from dataset.datasets_sharegpt4v import ShareGPT4VDataset
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

    query_dataset = ShareGPT4VDataset(
        image_data_path=args.image_data_path, 
        text_data_path=args.text_data_path,
        type='image',
    )

    cand_dataset = ShareGPT4VDataset(
        image_data_path=args.image_data_path, 
        text_data_path=args.text_data_path,
        type='text',
    )
    
    
    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process
    
    # debug --------------------------------------------------------------------------------------------
    rank0_print("模型初始化完成————————————————————————————————————————————————————————————————————————")
    rank0_print("mean_pooling: ",model.mean_pooling ,"use_bi_atten: ",model.use_bi_atten, "use_latent_atten: ",model.use_latent_atten)
    rank0_print("model.model.use_bi_atten: ", model.model.use_bi_atten)
    rank0_print("模型对应的类名：", model.__class__.__name__)
    rank0_print("实验脚本运行时间是：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rank0_print("args: ", args)
    rank0_print("query_dataset 使用的咒语： ",query_dataset.prompt)
    rank0_print("cand_dataset 使用的咒语： ",cand_dataset.prompt)
    rank0_print("-"*50)
    # ------------------------------------------------------------------------------------------------
    
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

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        # Adjust the order according to ids
        import numpy as np 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        image_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        text_features = candidate_features[sorted_candidate_indices]

        texts_image_index = [i for i in range(image_features.shape[0])] # 这是一个一维张量，表示每个文本对应的正样本图像的索引

        assert text_features.isnan().sum().item() == 0, 'nan in retrieve emb'
        assert image_features.isnan().sum().item() == 0, 'nan in images emb'

        # get the score for each text and image pair
        scores  = text_features @ image_features.t() # shape: (nb_texts, nb_images)

        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True  # 将 positive_pairs 中对应正样本的位置设置为 True。
        metrics = {}
        recall_k_list = [1, 5, 10]
        batch_size = 64
        for recall_k in recall_k_list:
            # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
            # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
            # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
            # for each image, that number will be greater than 1 for text retrieval.
            # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
            # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
            # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
            # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
            # it over the dataset.
            # image_retrieval_recall 是指图像检索的召回率，text_retrieval_recall 是指文本检索的召回率
            metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
            metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

        model_name = args.model_id.split('/')[-1]
        
        # debug 创建保存结果的目录 ------------------------------------------------------------
        save_dir_name = args.save_dir_name
        print("save_dir_name: ", save_dir_name)
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name, exist_ok=True)  # 递归创建目录，若存在则忽略
        # ----------------------------------------------------------------------------------

        with open(f"{save_dir_name}/{model_name}.txt", 'w') as f:
            f.write('sharegpt4v evaluation' + '\n')
            for key in metrics:
                f.write(f"{key} = {metrics[key]}" + "\n")
        print(metrics)

        # get data for rerank 
        if args.save_for_rerank:
            save_for_rerank_t2i(candidate_ids, query_ids, scores, save_dir_name)
            save_for_rerank_i2t(query_ids, candidate_ids, scores.T, save_dir_name)

def save_for_rerank_t2i(query_ids, candidate_ids, initial_scores, save_dir_name):
    index = []
    scores = []
    for i in range(len(initial_scores)):
        score = initial_scores[i:i+1]
        topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        index.append(topk_indexes)
        scores.append(topk_score.tolist())

    cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
    query_names = query_ids 

    with open(f"{save_dir_name}/sharegpt4v_t2i_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/sharegpt4v_t2i_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/sharegpt4v_t2i_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)

def save_for_rerank_i2t(query_ids, candidate_ids, initial_scores, save_dir_name):
    index = []
    scores = []
    for i in range(len(initial_scores)):
        score = initial_scores[i:i+1]
        topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
        topk_indexes = topk_indexes.squeeze().tolist()
        index.append(topk_indexes)
        scores.append(topk_score.tolist())

    cand_names = np.array([[candidate_ids[item] for item in row] for row in index])
    query_names = query_ids 

    with open(f"{save_dir_name}/sharegpt4v_i2t_query_names.json", 'w') as f:
        json.dump(query_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/sharegpt4v_i2t_cand_names.json", 'w') as f:
        json.dump(cand_names.tolist(), f, indent=2)
    with open(f"{save_dir_name}/sharegpt4v_i2t_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_path', type=str)
    parser.add_argument('--text_data_path', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_for_rerank', action='store_true')
    parser.add_argument('--save_dir_name', type=str, help='Directory name to save results')
    parser.add_argument('--use_instruction_mask', type=str, default='True', help='Whether to use instruction mask')
    parser.add_argument('--use_bi_atten', type=str, default='True', help='Whether to use bi attention')
    parser.add_argument('--mean_pooling', type=str, default='True', help='Whether to use mean pooling')
    args = parser.parse_args()
    eval(args)