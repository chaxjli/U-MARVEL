# Adapted from Tevatron code
import logging
import sys
import torch
import wandb

from transformers import (
    HfArgumentParser,
)

from src.dataset import TrainTextImageDataset
from src.collator import TrainTextImageDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name


logger = logging.getLogger(__name__)


def main():
    # 解决torch.distributed.launch参数解析问题
    # 参考: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    # 解析命令行参数到数据类
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 类型注解（不影响运行，提高代码可读性）
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # 初始化WandB日志记录（仅主进程执行）
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")

    # 构建模型（支持多模态，如图像-文本）
    model = MMEBModel.build(model_args, training_args)
    
    # 获取模型骨干网络名称（如bert-base-uncased）
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    
    # 加载模型处理器（分词器、图像处理器等）
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)

    # 初始化训练数据集（多模态文本-图像对）
    train_dataset = TrainTextImageDataset(data_args, model_args)
    
    # 初始化数据收集器（处理batch数据，如padding、图像增强）
    collator = TrainTextImageDataCollator(data_args, model_args, processor)

    # 使用梯度缓存训练器（支持大规模批处理和内存优化）
    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        max_length=data_args.max_len
    )
    
    # 将trainer引用传递给dataset（可能用于动态调整数据）
    train_dataset.trainer = trainer

    # 执行训练过程
    trainer.train()
    
    # 保存训练好的模型
    trainer.save_model(training_args.output_dir)

    # 仅在主进程中保存处理器
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()