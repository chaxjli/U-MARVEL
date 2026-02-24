# 该脚本已经实现 NPU 显存优化，将数据仅可能移动到 CPU 上面进行计算
import sys
import numpy as np
import os 
# 将上级目录添加到系统路径中，以便后续可以导入该目录下的模块
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

import json 
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from dataset.datasets_mbeir_eval_rerank_listwise import LazySupervisedDataset
import torch 
from tqdm import tqdm 
from collators.eval_rerank import EvalRerankDataCollator
from torch.utils.data import DataLoader 
from accelerate import Accelerator
import argparse 

# 定义重排序函数
def rerank(args):
    query_data_path = args.query_data_path 
    cand_pool_path = args.cand_pool_path 
    instructions_path = args.instructions_path
    model_id = args.model_id 
    original_model_id = args.original_model_id 
    ret_query_data_path = args.ret_query_data_path 
    ret_cand_data_path = args.ret_cand_data_path 
    image_path_prefix = args.image_path_prefix 
    rank_num = args.rank_num  
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer 

    # 从预训练模型中加载 Qwen2VL条件生成模型，使用 bfloat16 数据类型，减少 CPU 内存使用
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
    )
    # 将模型设置为评估模式
    model.eval()

    # 初始化 Accelerator，使用 bfloat16 混合精度
    accelerator = Accelerator(mixed_precision='bf16')
    # 获取当前设备
    device = accelerator.device 
    # 判断当前进程是否为主进程
    is_main_process = accelerator.is_main_process 

    # 将模型移动到指定设备
    model = model.to(device)

    # 初始化数据集
    dataset = LazySupervisedDataset(query_data_path, cand_pool_path, 
                                    instructions_path, ret_query_data_path, ret_cand_data_path, 
                                    image_path_prefix, rank_num=rank_num)
    # 初始化数据收集器
    data_collator = EvalRerankDataCollator(tokenizer=tokenizer, processor=processor)
    # 初始化数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=data_collator)

    # 将模型设置为评估模式
    model.eval()

    # 定义将张量移动到指定设备的函数
    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    # 存储所有的输出结果
    all_outputs = []
    # 存储所有的索引
    all_indexes = []

    # 使用 Accelerator 准备数据加载器和模型
    dataloader, model = accelerator.prepare(dataloader, model)

    # 遍历数据加载器中的每个批次
    for inputs, indexes in tqdm(dataloader):
        # 将输入数据移动到指定设备
        inputs = tensors_to_device(inputs, device)
        # 生成输出，同时获取得分和生成的字典 ------- debug
        with torch.npu.amp.autocast(): # 生成后立即释放 NPU 内存 debug ?????
            # 为什么作者模型需要 module 但是评估我们自己的模型 却不需要呢 ???????
            outputs = model.generate(**inputs, max_new_tokens=128, output_scores=True, 
                                        return_dict_in_generate=True, do_sample=False)
        # 获取生成的 ID 序列
        # 异步转移生成结果到CPU ----- debug
        with torch.npu.stream(torch.npu.Stream()):
            generated_ids = outputs.sequences.cpu()
        # 去除输入部分的 ID，只保留生成的部分
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        # 将生成的 ID 序列解码为文本
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 使用 Accelerator 收集所有进程的输出文本
        output_text = accelerator.gather_for_metrics(output_text)
        # 使用 Accelerator 收集所有进程的索引
        indexes = accelerator.gather_for_metrics(indexes)
        
        # 及时释放不再需要的变量 --- debug
        del outputs, generated_ids
        torch.npu.empty_cache()

        # 将当前批次的索引添加到所有索引列表中
        all_indexes.extend(indexes)
        
        # 将当前批次的输出文本添加到所有输出列表中
        all_outputs.extend(output_text)

    # 减少冗余
    index_set = set()
    filter_indexes = []
    filter_outputs = []

    # 如果是主进程
    if is_main_process:
        # 遍历所有索引和输出
        for idx, index in enumerate(all_indexes):
            if index in index_set:
                pass 
            else:
                index_set.add(index)
                filter_indexes.append(index)
                filter_outputs.append(all_outputs[idx])
        
        # 将过滤后的索引转换为 numpy 数组
        filter_indexes = np.array(filter_indexes) 
        # 获取排序后的索引
        sorted_filter_indices = np.argsort(filter_indexes)
        # 将过滤后的输出转换为 numpy 数组
        filter_outputs = np.array(filter_outputs)
        # 根据排序后的索引对输出进行排序
        filter_outputs = filter_outputs[sorted_filter_indices]

        # 存储所有查询ID
        query_ids = []
        # 存储查询ID到重排序输出的映射
        queryid2rerank_outputs = {}
        # 遍历数据集中的查询数据
        for item in dataset.query_data:
            query_ids.append(item['qid'])
        # 构建查询ID到重排序输出的映射
        for i, query_id in enumerate(query_ids):
            if query_id not in queryid2rerank_outputs:
                queryid2rerank_outputs[query_id] = filter_outputs[i]

        # 从命令行参数中获取保存结果的目录名
        save_dir_name = args.save_dir_name 
        # 如果保存目录不存在，则创建该目录
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)

        # 将查询 ID 到重排序输出的映射保存为 JSON 文件
        with open(f"{save_dir_name}/{args.save_name}_test_queryid2rerank_outputs_listwise.json", 'w') as f:
            json.dump(queryid2rerank_outputs, f, indent=2)

if __name__ == '__main__':
    
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data_path', type=str)
    parser.add_argument('--cand_pool_path', type=str)
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--ret_query_data_path', type=str)
    parser.add_argument('--ret_cand_data_path', type=str)
    parser.add_argument('--rank_num', type=int, default=10)  # 添加要排序的数量的命令行参数，默认值为10
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8) # 添加批次大小的命令行参数，默认值为8
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--save_dir_name', type=str)
    args = parser.parse_args()
    rerank(args)