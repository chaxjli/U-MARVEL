import os
import sys
import pathlib
import subprocess
import datetime

ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
os.environ["ASCEND_HOME_PATH"] = f'''{ASCEND_TOOLKIT_HOME}'''
os.environ["ASCEND_OPP_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/opp'''
os.environ["ASCEND_AICPU_PATH"] = f'''{ASCEND_TOOLKIT_HOME}'''
os.environ["TOOLCHAIN_HOME"] = f'''{ASCEND_TOOLKIT_HOME}/toolkit'''

ori_env_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/lib64/:{ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:{ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:{ori_env_path}'''
ori_env_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/tools/aml/lib64:{ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:{ori_env_path}'''

ori_env_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f'''{ASCEND_TOOLKIT_HOME}/python/site-packages:{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:{ori_env_path}'''


ori_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = f'''{ASCEND_TOOLKIT_HOME}/bin:{ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:{ori_env_path}'''

ori_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = f'''/home/user_name/miniconda3/envs/u-marvel/bin/:{ori_env_path}'''

try:
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    MASTER_PORT = 10060  # 主节点端口
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量
except Exception as e:
    print(f"Error: {e}")
    print("多机环境加载失败，使用单机单卡训练")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['NNODES'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['NPROC_PER_NODE'] = '8'
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    MASTER_PORT = 10060  # 主节点端口
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量

# 检查分布式参数是否设置
required_vars = ['MASTER_ADDR', 'NNODES', 'NODE_RANK', 'NPROC_PER_NODE']
for var in required_vars:
    if not os.getenv(var):
        print(f"Error: {var} must be set as an environment variable.")
        sys.exit(1)

DISTRIBUTED_ARGS = f"--nnodes {NNODES} --nproc_per_node {NPROC_PER_NODE} --node_rank {NODE_RANK} " \
                   f"--master_addr {MASTER_ADDR} --master_port {MASTER_PORT} --rdzv_backend c10d " \
                   f"--rdzv_endpoint {MASTER_ADDR}:{MASTER_PORT}"
MODEL_ID = "qwen2-vl-7b"
# 过滤 filt
FIRST_MODEL_NAME = "qwen2-vl-7b_umarvel_progressive_transition_mmeb"
RUN_ID = f"{FIRST_MODEL_NAME}-Rank-Only-Pointwise"
MODEL_LOCAL_PATH = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
QUERY_DATA_PATH = "./data/MMEB-train/mmeb_union_query.jsonl"
CAND_POOL_PATH = "./data/MMEB-train/mmeb_union_candidate.jsonl"
INSTRUCTIONS_PATH = "./data/MMEB-train/data_instruction.json"  # 指令数据集
RERANK_DATA_PATH = f"./result/mmeb/train/{FIRST_MODEL_NAME}_eval_results_finetune/local/rerank_data_all_eval_train_local.json"
IMAGE_PATH_PREFIX = "./data/MMEB-train"  # 图片路径前缀

TRAIN_VISION_ENCODER = False
USE_VISION_LORA = False
TRAIN_VISION_PROJECTOR = False

USE_LORA = True
Q_LORA = False
LORA_R = 128
LORA_ALPHA = 256

DS_STAGE = "ds_config_zero1"
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 1
LR = 2e-5
MODEL_MAX_LEN = 2048

# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 创建结果目录
os.makedirs(f"./checkpoints/rerank_model/mmeb/{RUN_ID}", exist_ok=True)
total_log_file = f"./checkpoints/rerank_model/mmeb/{RUN_ID}/{TIMENAME}total_stdout.log"
original_stdout = sys.stdout
sys.stdout = open(total_log_file, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印模型的参数信息
print("-"*50)
print("MODEL_ID",MODEL_ID)
print("RUN_ID",RUN_ID)
print("MODEL_LOCAL_PATH",MODEL_LOCAL_PATH)
print("FIRST_MODEL_NAME",FIRST_MODEL_NAME)
print(required_vars,[MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])


# ==== 启动训练 ====
train_command = [
    "torchrun",
    *DISTRIBUTED_ARGS.split(),
    "train/mmeb/train_rerank_multi_nodes_only_pointwise.py",
    "--model_id", MODEL_ID,
    "--query_data_path", QUERY_DATA_PATH,
    "--cand_pool_path", CAND_POOL_PATH,
    "--instructions_path", INSTRUCTIONS_PATH,
    "--output_dir", f"./checkpoints/rerank_model/mmeb/{RUN_ID}",
    "--report_to", "tensorboard",
    "--run_name", RUN_ID,
    "--deepspeed", f"./ds_configs/{DS_STAGE}.json",
    "--bf16", "True",
    "--num_train_epochs", str(NUM_EPOCHS),
    "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--per_device_eval_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
    "--eval_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "2000",
    "--save_total_limit", "20",
    "--learning_rate", str(LR),
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "False",
    "--model_max_length", str(MODEL_MAX_LEN),
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",
    "--train_vision_encoder", str(TRAIN_VISION_ENCODER),
    "--use_vision_lora", str(USE_VISION_LORA),
    "--train_vision_projector", str(TRAIN_VISION_PROJECTOR),
    "--use_lora", str(USE_LORA),
    "--q_lora", str(Q_LORA),
    "--lora_r", str(LORA_R),
    "--lora_alpha", str(LORA_ALPHA),
    "--model_local_path", MODEL_LOCAL_PATH,
    "--rerank_data_path", RERANK_DATA_PATH,
    "--image_path_prefix", IMAGE_PATH_PREFIX
]

try:
    # 打开总的日志文件以追加模式写入
    sys.stdout.flush()  # 强制刷新缓冲区
    with open(total_log_file, 'a') as log_file:
        print("训练命令:", train_command)
        # subprocess.run(train_command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        subprocess.run(train_command, stdout=log_file, stderr=None, check=True)
    print(f"任务执行完成，日志已记录到 {total_log_file}")
except subprocess.CalledProcessError as e:
    print(f"任务执行失败: {e}")


# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout





