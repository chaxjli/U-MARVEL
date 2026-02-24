import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen3_vl_finetune import Qwen3VLRetFinetuneForConditionalGeneration
import torch
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
import argparse
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
from peft import PeftModel 
import shutil 

def eval(args):
    accelerator = Accelerator()
    original_model_id = args.original_model_id
    model_id = args.model_id 

    device = accelerator.device  # 使用 Accelerator 的设备
    if accelerator.is_main_process:
        print(f"Using device: {device}")

    # 加载原始模型，并迁移到设备
    model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(
        original_model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True, 
    )
    model.to(device)

    # 加载 LoRA 模型，并迁移到设备
    lora_model = PeftModel.from_pretrained(model, model_id)
    lora_model.to(device)

    # 合并 LoRA 权重
    merged_model = lora_model.merge_and_unload()
    merged_model.to('cpu')  # 移动到 CPU，以便保存

    # 加载处理器
    processor = AutoProcessor.from_pretrained(original_model_id)

    # 只有主进程保存模型和处理器
    if accelerator.is_main_process:
        try:
            merged_model.save_pretrained(args.save_path)
            accelerator.print(f"合并后的模型保存到： {args.save_path}. 文件列表:")
            accelerator.print(os.listdir(args.save_path))
        except Exception as e:
            accelerator.print(f"Error saving model: {e}")
            raise
        processor.save_pretrained(args.save_path)

        # 复制 chat_template.json 文件
        source_chat_file = os.path.join(args.original_model_id, "chat_template.json")
        target_chat_file = os.path.join(args.save_path, "chat_template.json")
        if os.path.exists(source_chat_file):
            shutil.copy(source_chat_file, target_chat_file)
        else:
            accelerator.print(f"{source_chat_file} does not exist.")
        # 保存 sim.bin 文件
        if "sim.bin" in os.listdir(args.model_id):
            source_sim_file = os.path.join(args.model_id, "sim.bin")
            target_sim_file = os.path.join(args.save_path, "sim.bin")
            shutil.copy(source_sim_file, target_sim_file)

    # 等待所有进程完成
    accelerator.wait_for_everyone()

    # 主进程验证模型是否能够重新加载
    if accelerator.is_main_process:
        try:
            test_model = Qwen3VLRetFinetuneForConditionalGeneration.from_pretrained(args.save_path) 
            accelerator.print("Model reloaded successfully.")
            accelerator.print("温度参数:", test_model.sim.temp)
        except Exception as e:
            accelerator.print(f"Error reloading model: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    eval(args)