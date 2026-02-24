"""
多模态模型微调脚本，支持 LoRA 和 QLoRA，适用于 NPU 加速。
主要功能：加载模型与数据，配置训练参数，进行高效微调，并保存结果。
"""

import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)  # 添加自定义模块路径

from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml  # 用于 YAML 配置文件处理

# 深度学习相关库
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # 参数高效微调工具
import torch
import torch_npu  # 适配 NPU
from torch_npu.contrib import transfer_to_npu  # 将张量转移到 NPU

# Transformers库组件
import transformers
from transformers import Trainer, deepspeed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader

# 自定义模块
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments  # 训练参数配置
from collators.mbeir_rerank import MbeirRerankDataCollator                             # 数据整理器
from dataset.datasets_mmeb_rerank_only_pointwise import LazySupervisedDataset  # 惰性加载数据集
from loaders.qwen2_vl_raw import Qwen2VLModelLoader              # 模型加载器
from supported_models import MODULE_KEYWORDS                     # 模型结构关键字
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3                                  # 工具函数
)

def train():
        
    """主训练流程，包含参数解析、模型加载、数据准备、训练配置等步骤"""
    # 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # 在 train() 函数中获取全局 rank 和 world size -------------------------------
    if dist.is_initialized():
        global global_rank, world_size, local_rank
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 参数持久化 ----------------------------------------------------------------
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "必须指定输出目录"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    # 计算精度配置 --------------------------------------------------------------
    compute_dtype = (
        torch.float16 if training_args.fp16 
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    # 分布式训练配置检查
     # 如果使用了 deepspeed 且启用了 QLoRA
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # 设备映射配置（ QLoRA 模式需要特殊处理）
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP 或 ZeRO3 与 QLoRA 不兼容")

    # QLoRA 量化配置 ------------------------------------------------------------
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("启用 LLM 量化配置...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                     # 4bit量化加载
            bnb_4bit_compute_dtype=compute_dtype,  # 计算时数据类型
            bnb_4bit_quant_type="nf4",             # 量化类型
        )
    
    # 加载模型/分词器/处理器 ----------------------------------------------------
    rank0_print("加载模型、分词器、处理器...")
    loader = Qwen2VLModelLoader(
        model_hf_path=model_args.model_hf_path,        # HuggingFace模型路径
        model_local_path=model_args.model_local_path,  # 本地模型路径
        compute_dtype=compute_dtype,                   # 计算精度
        bnb_config=bnb_config,                         # 量化配置
        use_flash_attn=training_args.use_flash_attn,   # 是否使用 FlashAttention
        device_map=device_map,                         # 设备映射
    )
    model, tokenizer, processor = loader.load()        # 加载三件套
    tokenizer.model_max_length = training_args.model_max_length  # 设置分词器最大长度

    # 梯度检查点配置
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # 启用模型的梯度检查点功能

    # 冻结视觉编码器参数 --------------------------------------------------------
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print(f"冻结视觉编码器参数，包含模块:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)  # 动态执行冻结操作

    # 冻结视觉投影器参数 --------------------------------------------------------
    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"冻结视觉投影器参数，包含模块:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False) # 动态执行冻结操作，将指定模块的参数设置为不可训练

    # 冻结其他多模态组件 --------------------------------------------------------
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"冻结其他多模态组件:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # LoRA配置部分 -------------------------------------------------------------
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        rank0_print("未启用LoRA...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}  
        lora_modules = [] # 应用 LoRA 的模块列表
        full_modules = [] # 全参数训练的模块列表

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("启用视觉编码器LoRA...")
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
            
        elif training_args.train_vision_encoder:
            rank0_print("视觉编码器将全参数训练...")
            full_modules.extend(vision_encoder_keys)
        if lora_args.use_lora:
            rank0_print("启用 LLM 的 LoRA...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            rank0_print("LLM 将全参数训练...")
            full_modules.extend(llm_keys)
        if training_args.train_vision_projector:
            rank0_print("视觉投影器将全参数训练...")
            full_modules.extend(vision_projector_keys)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,                     # LoRA秩
            lora_alpha=lora_args.lora_alpha,        # 缩放系数
            target_modules=lora_modules,            # 目标模块列表
            modules_to_save=full_modules,           # 全参数训练模块
            lora_dropout=lora_args.lora_dropout,    # Dropout率
            bias=lora_args.lora_bias,               # 偏置项处理方式
            task_type="CAUSAL_LM",                  # 任务类型
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        
    # 打印可训练参数 ------------------------------------------------------------
    rank0_print("可训练参数列表:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # 数据加载部分 --------------------------------------------------------------
    rank0_print("加载训练数据...")
    train_dataset = LazySupervisedDataset(
        query_data_path=data_args.query_data_path,  # 查询数据路径
        cand_pool_path=data_args.cand_pool_path,    # 候选池路径
        instructions_path=data_args.instructions_path,  # 指令文件路径
        rerank_data_path=data_args.rerank_data_path,    # 重排数据路径
        image_path_prefix=data_args.image_path_prefix,  # 图像路径前缀
        tokenizer=tokenizer,            # 分词器
    )
    rank0_print("数据加载完成————————————————————————————————————————————————")
    # 数据整理器配置
    data_collator = MbeirRerankDataCollator(
        tokenizer=tokenizer,
        processor=processor,  # 图像处理器
        # is_reset_max_pixels=False,  # 是否重置最大像素值
    )
    model.floating_point_ops = lambda s: 0  # 禁用浮点运算计数（节省资源）
    trainer = Trainer(
        model=model,
        args=training_args,           # 训练参数
        data_collator=data_collator,  # 数据整理器
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=None,            # 无验证集
    )
    # 训练流程
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)

    
if __name__ == "__main__":
    train()