import os
import sys
# 获取当前文件的绝对路径所在的目录
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 构建上级目录的路径
module_path = os.path.join(current_file_path, "../")
# 将上级目录添加到 Python 模块搜索路径中
sys.path.append(module_path)
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch_npu  # 导入华为昇腾 NPU 相关的 PyTorch 扩展
from torch_npu.contrib import transfer_to_npu # 适配 npu

import transformers
from transformers import Trainer, deepspeed

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from dataset.datasets_nli import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3
)


def train():
    # 创建一个 HfArgumentParser 对象，用于解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    # 将命令行参数解析为对应的 dataclass 对象
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    # 获取训练参数中的输出目录
    output_dir = getattr(training_args, 'output_dir', None)
    # 确保输出目录存在
    assert output_dir is not None, "output_dir is required"
    # 构建保存参数的目录
    args_dir = Path(output_dir) / "arguments"
    # 创建保存参数的目录，如果父目录不存在则一并创建
    args_dir.mkdir(parents=True, exist_ok=True)
    # 将模型参数保存为 YAML 文件
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    # 将数据参数保存为 YAML 文件
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    # 将训练参数保存为 YAML 文件
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    # 将 LoRA 参数保存为 YAML 文件
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    # 根据训练参数中的精度设置确定计算数据类型
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # 如果使用 DeepSpeed 且启用了 Q-LoRA
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        # 设置分布式训练类型为 DeepSpeed
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # 设备映射，用于指定模型在哪些设备上运行
    device_map = None
    # 如果启用了 Q-LoRA
    if lora_args.q_lora:
        # 如果是分布式训练，根据环境变量设置设备映射
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        # 如果同时使用了 FSDP 或 ZeRO3，抛出错误
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # llm quantization config (for q-lora)
    # 用于 Q-LoRA 的量化配置
    bnb_config = None
    # 如果使用 LoRA 且启用了 Q-LoRA
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        # 打印信息，表示启用了大语言模型的量化
        rank0_print("Quantization for LLM enabled...")
        # 配置量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 以 4 位量化加载模型
            bnb_4bit_compute_dtype=compute_dtype,  # 4 位量化的计算数据类型
            bnb_4bit_quant_type="nf4",  # 4 位量化类型为 nf4
        )
    
    # load model, tokenizer, processor
    # 打印信息，表示正在加载模型、分词器和处理器
    rank0_print("Loading model, tokenizer, processor...")
    # 根据模型家族 ID 选择对应的加载器
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,  # Hugging Face 模型路径
        model_local_path=model_args.model_local_path,  # 本地模型路径
        compute_dtype=compute_dtype,  # 计算数据类型
        bnb_config=bnb_config,  # 量化配置
        use_flash_attn=training_args.use_flash_attn,  # 是否使用 Flash Attention
        device_map=device_map,  # 设备映射
    )
    # 加载模型、分词器和处理器
    model, tokenizer, processor = loader.load()
    # 设置分词器的最大输入长度
    tokenizer.model_max_length = training_args.model_max_length

    # 如果启用了梯度检查点
    if training_args.gradient_checkpointing:
        # 启用模型的输入梯度计算
        model.enable_input_require_grads()

"""
MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "qwen2-vl-7b": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    },
    "qwen2-vl-2b": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    }
}
"""

    # freeze certain params
    # 获取模型家族对应的视觉编码器模块关键字
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    # 如果不训练视觉编码器
    if not training_args.train_vision_encoder:
        # 打印信息，表示视觉编码器已冻结
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            # 打印冻结的模块名称
            rank0_print(f"\t{module}")
            # 冻结该模块的参数，不进行梯度更新
            eval(f"model.{module}").requires_grad_(False)

    # 获取模型家族对应的视觉投影器模块关键字
    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    # 如果不训练视觉投影器
    if not training_args.train_vision_projector:
        # 打印信息，表示视觉投影器已冻结
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            # 打印冻结的模块名称
            rank0_print(f"\t{module}")
            # 冻结该模块的参数，不进行梯度更新
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # 如果模型家族配置中包含其他多模态组件
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        # 打印信息，表示其他多模态组件已冻结
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            # 打印冻结的组件名称
            rank0_print(f"\t{other_key}")
            # 冻结该组件的参数，不进行梯度更新
            eval(f"model.{other_key}").requires_grad_(False)

    # lora preparation
    # 获取模型家族对应的大语言模型模块关键字
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    # 如果不使用 LoRA 或者 （不训练视觉编码器且不使用视觉 LoRA）
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        # 打印信息，表示未启用 LoRA
        rank0_print("No LoRA enabled...")        
    else:
        # 获取模型的所有命名模块
        named_modules = {n: m for n, m in model.named_modules()}
        # 存储需要应用 LoRA 的模块名称
        lora_modules = []
        # 存储需要完全训练的模块名称
        full_modules = []

        # 如果训练视觉编码器且使用视觉 LoRA
        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            # 打印信息，表示启用了视觉编码器的 LoRA
            rank0_print("LoRA for vision encoder enabled...")

            # 找到视觉编码器中所有需要应用 LoRA 的线性层名称
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
            
        # 如果只训练视觉编码器
        elif training_args.train_vision_encoder:
            # 打印信息，表示视觉编码器将被完全训练
            rank0_print("Vision encoder will be fully trained...")
            # 将视觉编码器模块添加到完全训练的模块列表中
            full_modules.extend(vision_encoder_keys)
        
        # 如果使用 LoRA
        if lora_args.use_lora:
            # 打印信息，表示启用了大语言模型的 LoRA
            rank0_print("LoRA for LLM enabled...")
            # 找到大语言模型中所有需要应用 LoRA 的线性层名称
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            # 打印信息，表示大语言模型将被完全训练
            rank0_print("LLM will be fully trained...")
            # 将大语言模型模块添加到完全训练的模块列表中
            full_modules.extend(llm_keys)
        
        # 如果训练视觉投影器
        if training_args.train_vision_projector:
            # 打印信息，表示视觉投影器将被完全训练
            rank0_print("Vision projector will be fully trained...")
            # 将视觉投影器模块添加到完全训练的模块列表中
            full_modules.extend(vision_projector_keys)
        
        # 配置 LoRA 参数
        lora_config = LoraConfig(
            r=lora_args.lora_r,  # LoRA 矩阵的秩
            lora_alpha=lora_args.lora_alpha,  # LoRA 的缩放因子
            target_modules=lora_modules,  # 需要应用 LoRA 的目标模块
            modules_to_save=full_modules,  # 需要完全训练并保存的模块
            lora_dropout=lora_args.lora_dropout,  # LoRA 的丢弃率
            bias=lora_args.lora_bias,  # 是否训练偏置项
            task_type="CAUSAL_LM",  # 任务类型为因果语言模型
        )

        # 如果启用了 Q-LoRA
        if lora_args.q_lora:
            # 为模型准备 K 位训练
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
        # 将 LoRA 配置应用到模型上
        model = get_peft_model(model, lora_config)
        
    # print trainable parameters for inspection
    # 打印可训练的参数信息，用于检查
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 打印可训练的参数名称
            rank0_print(f"\t{name}")

    # load data
    # 打印信息，表示正在加载数据
    rank0_print("Loading data...")
    # 加载训练数据集
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,  # 数据集路径
    )
    
    # 评估数据集初始化为 None
    eval_dataset = None
    # 设置评估策略为不进行评估
    training_args.eval_strategy = "no"

    # data collator
    # 根据模型家族 ID 选择对应的数据收集器
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,  # 分词器
        processor=processor,  # 处理器
    )

    # 添加梯度检查点的配置参数
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False} 
    # 创建 Trainer 对象，用于模型训练
    trainer = Trainer(
        model=model,  # 待训练的模型
        args=training_args,  # 训练参数
        data_collator=data_collator,  # 数据收集器
        train_dataset=train_dataset,  # 训练数据集
    )
    
    # 开始训练模型
    trainer.train()
    # 保存训练状态
    trainer.save_state()

    # 安全地保存模型
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    

if __name__ == "__main__":
    # 调用 train 函数开始训练
    train()