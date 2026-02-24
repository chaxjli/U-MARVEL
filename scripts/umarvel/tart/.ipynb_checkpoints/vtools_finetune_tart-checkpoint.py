import os
import sys
import pathlib
import subprocess
import datetime
# ==== NPU 环境配置 ====
# 激活虚拟环境, 这样激活虚拟环境只会使得子模块在该虚拟环境当中进行，主模块仍然在原来的环境当中运行
# os.system("source /home/user_name/miniconda3/bin/activate && conda activate u-marvel")
# os.environ["ASCEND_LAUNCH_BLOCKING"]="1"  # debug 时才开启，明白了

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


# 设置分布式环境变量 ----- 提交 vtools 时注销,单机训练不需要注销
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['NNODES'] = '1'
os.environ['NODE_RANK'] = '0'
os.environ['NPROC_PER_NODE'] = '1' # 单机单卡 debug 训练时设置为1


# ==== 分布式参数 ====
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

# ==== 分布式启动参数 ====
DISTRIBUTED_ARGS = f"--nnodes {NNODES} --nproc_per_node {NPROC_PER_NODE} --node_rank {NODE_RANK} " \
                   f"--master_addr {MASTER_ADDR} --master_port {MASTER_PORT} --rdzv_backend c10d " \
                   f"--rdzv_endpoint {MASTER_ADDR}:{MASTER_PORT}"


# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 定义参数
MODEL_ID = "qwen2-vl-7b"
QUERY_DATA_PATH = "./data/M-BEIR/query/union_train/mbeir_union_up_train.jsonl"
CAND_POOL_PATH = "./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl"
INSTRUCTIONS_PATH = "./data/M-BEIR/instructions/query_instructions.tsv"
MODEL_LOCAL_PATH = "./checkpoints/BiLamRA_Ret_PretrainWithMeanpooling_Withoutprompt_token_cls-merged"
IMAGE_PATH_PREFIX = "./data/M-BEIR"
QUERY_DATASET_HAS_INSTRUCTION = True              # 是否做指令匹配
QUERY_DATASET_USE_INSTRUCTION_TOKEN = False       # 是否使用指令 token
MODEL_MEAN_POOLING = True                         # 是否使用全局平局池化
MODEL_USE_BI_ATTEN = True                         # 是否使用双向注意力模块
MODEL_USE_LATENT_ATTEN = False                    # 是否使用潜在注意力模块
MODEL_USE_INSTRUCTION_MASK = True                 # 是否使用指令 mask

# 定义训练参数
TRAIN_VISION_ENCODER = False
USE_VISION_LORA = False
TRAIN_VISION_PROJECTOR = False
USE_LORA = True
Q_LORA = False
LORA_R = 128
LORA_ALPHA = 256
RUN_ID = f"{MODEL_ID}_BiLamRA_Ret_mask_instruction_token_test"
DS_STAGE = "ds_config_zero1"
PER_DEVICE_BATCH_SIZE = 60
GRAD_ACCUM = 1
NUM_EPOCHS = 1
LR = 1e-4
MODEL_MAX_LEN = 1024

# 创建结果目录
os.makedirs(f"./checkpoints/{RUN_ID}", exist_ok=True)
# 创建日志文件位置
log_file_path = f"./checkpoints/{RUN_ID}/{TIMENAME}stdout.log"
# 重定向标准输出到日志文件
original_stdout = sys.stdout
sys.stdout = open(log_file_path, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印参数
print("模型ID: ",MODEL_ID)
print("模型名称: ",RUN_ID)
print("预训练使用的模型路径: ", MODEL_LOCAL_PATH)
print("MASTER_ADDR: ",MASTER_ADDR)
print("NNODES: ",NNODES)
print("NODE_RANK: ",NODE_RANK)
print("NPROC_PER_NODE: ",NPROC_PER_NODE)
print("DISTRIBUTED_ARGS: ",DISTRIBUTED_ARGS)
print("TIMENAME: ",TIMENAME)

# 构建 torchrun 命令
torchrun_command = [
    "torchrun",
    *DISTRIBUTED_ARGS.split(),
    "train/train_mbeir.py",
    "--model_id", MODEL_ID,
    "--query_data_path", QUERY_DATA_PATH,
    "--cand_pool_path", CAND_POOL_PATH,
    "--instructions_path", INSTRUCTIONS_PATH,
    "--output_dir", f"./checkpoints/{RUN_ID}",
    "--report_to", "tensorboard",
    "--run_name", RUN_ID,
    "--deepspeed", f"./ds_configs/{DS_STAGE}.json",
    "--bf16", "True",
    "--num_train_epochs", str(NUM_EPOCHS),
    "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--per_device_eval_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
    "--eval_strategy", "epoch",
    "--save_strategy", "epoch",
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
    "--image_path_prefix", IMAGE_PATH_PREFIX,
    "--use_instruction_token", str(QUERY_DATASET_USE_INSTRUCTION_TOKEN),
    "--has_instruction", str(QUERY_DATASET_HAS_INSTRUCTION),
    "--mean_pooling", str(MODEL_MEAN_POOLING),
    "--use_bi_atten", str(MODEL_USE_BI_ATTEN),
    "--use_latent_atten", str(MODEL_USE_LATENT_ATTEN),
    "--use_instruction_mask", str(MODEL_USE_INSTRUCTION_MASK),
]

# 将命令列表转换为字符串
command_str = " ".join(torchrun_command)

# 构建重定向输出到日志文件的命令
redirect_command = f"{command_str} >> {log_file_path}"
try:
    # 运行命令
    sys.stdout.flush()  # 强制刷新缓冲区
    result = os.system(redirect_command)
    if result != 0:
        print(f"命令执行失败，返回码: {result}")
    else:
        print("命令执行成功")
except Exception as e:
    print(f"执行命令时出现异常: {e}")

# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout