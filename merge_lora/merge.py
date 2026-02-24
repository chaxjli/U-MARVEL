import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
import torch
# import torch_npu                              # 适配 npu
# from torch_npu.contrib import transfer_to_npu # 适配 npu
import argparse
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
from peft import PeftModel 
import shutil 
def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        original_model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, 
    )
    lora_model = PeftModel.from_pretrained(model, model_id)
    merged_model = lora_model.merge_and_unload()
    # merged_model = merged_model.to('cpu')  # 移动到 CPU 以便保存
    
    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)


    # merged_model.save_pretrained
    try:
        merged_model.save_pretrained(args.save_path)
        print(f"合并后的模型保存到： {args.save_path}. Files:")
        print(os.listdir(args.save_path))
    except Exception as e:
        print(f"Error saving model: {e}")
        raise
    
    # merged_model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)

    # copy the chat_template.json file
    source_chat_file = os.path.join(args.original_model_id, "chat_template.json")
    target_chat_file = os.path.join(args.save_path, "chat_template.json")
    shutil.copy(source_chat_file, target_chat_file)
    
    # 验证模型是否能够重新加载
    try:
        test_model = Qwen2VLRetForConditionalGeneration.from_pretrained(args.save_path)
        print("Model reloaded successfully.")
    except Exception as e:
        print(f"Error reloading model: {e}")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    eval(args)