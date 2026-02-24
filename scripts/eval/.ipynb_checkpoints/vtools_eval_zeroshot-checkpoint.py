import subprocess
import logging

# 配置日志记录
logging.basicConfig(filename='eval_urban1k.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 定义模型 ID
ORIGINAL_MODEL_ID = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID = "code-kunkun/LamRA-Ret"

# 构建命令
command = [
    "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'",
    "accelerate",
    "launch",
    "--multi_gpu",
    "--main_process_port",
    "29508",
    "eval/eval_zeroshot/eval_urban1k.py",
    "--image_data_path", "./data/Urban1k/image",
    "--text_data_path", "./data/Urban1k/caption",
    "--batch_size", "4",
    "--original_model_id", ORIGINAL_MODEL_ID,
    "--model_id", MODEL_ID,
    "--save_for_rerank"
]

# 将命令列表转换为字符串
command_str = " ".join(command)

# 记录要执行的命令到日志
logging.info(f"即将执行的命令: {command_str}")

try:
    # 执行命令
    result = subprocess.run(command_str, shell=True, check=True, text=True, capture_output=True)
    # 记录命令执行成功信息和标准输出到日志
    logging.info("命令执行成功。")
    logging.info(f"命令输出:\n{result.stdout}")
except subprocess.CalledProcessError as e:
    # 记录命令执行失败信息和标准错误到日志
    logging.error(f"命令执行失败，错误信息:\n{e.stderr}")
