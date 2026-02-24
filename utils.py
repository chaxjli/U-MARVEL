import math
from typing import List, Dict, Optional

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length


class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def rank0_print_nested_dict(data, indent=0, level=0):
    """
    递归打印嵌套字典结构，并显示每个键的子元素数量
    :param data: 输入数据（字典、列表或元组）
    :param indent: 初始缩进量（默认0）
    :param level: 当前层级（默认0）
    """
    if isinstance(data, dict):
        for key, value in data.items():
            # 计算子元素数量
            sub_count = len(value) if isinstance(value, (dict, list, tuple)) else 0
            
            # 打印当前层信息
            prefix = '  ' * level
            rank0_print(f"{prefix}Key: {key} ({sub_count} sub-elements)")
            
            # 递归处理下一层
            rank0_print_nested_dict(value, indent, level + 1)
    
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list, tuple)):
                # 打印列表/元组索引信息
                prefix = '  ' * (level - 1)
                print(f"{prefix}[Index {idx}] ({len(item) if isinstance(item, (dict, list, tuple)) else 0} sub-elements)")
                rank0_print_nested_dict(item, indent, level)
import json

def read_json_file(file_path):
    """
    读取指定路径的 JSON 文件并返回其内容。

    :param file_path: JSON 文件的路径
    :return: 解析后的 JSON 数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"无法解析 {file_path} 中的 JSON 数据。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    return None

import torch
from torch import nn
from collections import OrderedDict
def rank0_print_module_memory(
    module: nn.Module,
    prefix: str = "",
    is_last: bool = False,
    parent_prefix: str = ""
):
    """
    完全递归打印所有子模块显存占用（无限深度）
    
    Args:
        module (nn.Module): 当前模块
        prefix (str): 当前模块名前缀
        is_last (bool): 是否当前层最后一个模块
        parent_prefix (str): 父层前缀（用于生成连接线）
    """
    # 统计当前模块参数显存
    param_mem = 0.0
    grad_mem = 0.0
    for p in module.parameters(recurse=True):
        param_mem += p.numel() * p.element_size() / 1024**2
        if p.grad is not None:
            grad_mem += p.grad.numel() * p.grad.element_size() / 1024**2

    # 生成树状前缀
    if parent_prefix == "":
        branch = "【Root】"
    else:
        branch = "├──" if not is_last else "└──"
    
    # 打印当前模块信息
    mem_info = ""
    if param_mem > 0 or grad_mem > 0:
        mem_info = f": {param_mem + grad_mem:.2f} MB (param: {param_mem:.2f} | grad: {grad_mem:.2f})"
    
    rank0_print(f"{parent_prefix}{branch} 【{prefix}】{mem_info}")

    # 递归子模块
    children = list(module.named_children())
    for idx, (name, child) in enumerate(children):
        is_last_child = idx == len(children)-1
        new_parent_prefix = parent_prefix + ("    " if is_last else "│  ")
        
        rank0_print_module_memory(
            child,
            prefix=name,
            is_last=is_last_child,
            parent_prefix=new_parent_prefix
        )

def rank0_print_model_memory_summary(model: nn.Module):
    """打印模型显存摘要（完全递归）"""
    rank0_print("\n=== Model Memory Summary ===")
    
    # 计算总显存
    total_mem = 0.0
    for p in model.parameters():
        total_mem += p.numel() * p.element_size() / 1024**2
        if p.grad is not None:
            total_mem += p.grad.numel() * p.grad.element_size() / 1024**2
    
    rank0_print(f"[Total] {total_mem:.2f} MB")
    rank0_print_module_memory(model)
