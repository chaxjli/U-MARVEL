import os
import sys
import pathlib
import subprocess
import datetime

# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# ==== NPU 环境配置 ====
# 激活虚拟环境, 这样激活虚拟环境只会使得子模块在该虚拟环境当中进行，主模块仍然在原来的环境当中运行
# os.system("source /home/user_name/miniconda3/bin/activate && conda activate u-marvel_qwen3")
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
os.environ["PATH"] = f'''/home/user_name/miniconda3/envs/u-marvel_qwen3/bin/:{ori_env_path}'''

try:
    # ==== 分布式参数 ====
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    MASTER_PORT = 10060  # 主节点端口
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量
except Exception as e:
    print(f"Error: {e}")
    # 设置分布式环境变量 ----- 提交 vtools 时注销,单机训练不需要注销
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['NNODES'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['NPROC_PER_NODE'] = '8'
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

MODEL_ID = "qwen3-vl-8b"
TRAIN_DATA_PATH = "./data/nli_for_simcse.csv"
RUN_ID = f"{MODEL_ID}_nli_pertrain_stage1_model"
EVAL_DATA_PATH = None
QUERY_DATASET_HAS_INSTRUCTION = False              # 是否做指令匹配
QUERY_DATASET_USE_INSTRUCTION_TOKEN = False        # 是否使用指令 token
HAS_HARD_NEGATIVE = False                          # 是否使用 hard negative
HAS_MODALITY_HARD_NEGATIVE = False                 # 是否使用 modality hard negative
MODEL_USE_BI_ATTEN = True                          # 是否使用双向注意力模块

# 注意：全局平均池化和自注意力池化不能同时使用
MODEL_MEAN_POOLING = True                          # 是否使用全局平局池化
MODEL_USE_SELF_ATTENT_POOLING = False              # 是否使用自注意力池化

MODEL_USE_LATENT_ATTEN = False                     # 是否使用潜在注意力模块
MODEL_USE_INSTRUCTION_MASK = False                 # 是否使用指令 mask
MODEL_USE_BI_LOSS = True                           # 是否使用双向损失
MODEL_USE_ISOTROPY_LOSS = False                    # 是否使用各向同性损失

# 注意：交叉熵损失和，FocalInfoNCELoss，DIHT_LOSS 不能同时使用，有且仅有一个可以为 True
MODEL_USE_CROSS_ENTROPY_LOSS = True                # 是否使用交叉熵损失
MODEL_USE_FOCAL_INFONCE_LOSS = False               # 是否使用 FocalInfoNCELoss
MODEL_USE_DIHT_LOSS = False                        # 是否使用 DIHTLoss

TRAIN_VISION_ENCODER = False
USE_VISION_LORA = False
TRAIN_VISION_PROJECTOR = False
USE_LORA = True
Q_LORA = False
LORA_R = 64
LORA_ALPHA = 128


DS_STAGE = "ds_config_zero1"
PER_DEVICE_BATCH_SIZE = 72
GRAD_ACCUM = 1
NUM_EPOCHS = 2

LR = 2e-4
MODEL_MAX_LEN = 1024

# 创建输出目录
output_dir = f"./checkpoints/{RUN_ID}"
os.makedirs(output_dir, exist_ok=True)
# 创建日志文件位置
log_file_path = f"./checkpoints/{RUN_ID}/{TIMENAME}stdout.log"
# 重定向标准输出到日志文件
original_stdout = sys.stdout
sys.stdout = open(log_file_path, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印参数
print("模型ID: ",MODEL_ID)
print("模型名称: ",RUN_ID)
print("数据集路径: ",TRAIN_DATA_PATH)
print("输出目录: ",output_dir)
print("MASTER_ADDR: ",MASTER_ADDR)
print("NNODES: ",NNODES)
print("NODE_RANK: ",NODE_RANK)
print("NPROC_PER_NODE: ",NPROC_PER_NODE)
print("DISTRIBUTED_ARGS: ",DISTRIBUTED_ARGS)
print("TIMENAME: ",TIMENAME)

# 构建 torchrun 命令
torchrun_command = ["torchrun"] + [*DISTRIBUTED_ARGS.split()]+ [
    "train/train_nli_qwen3_vl.py",
    "--model_id", MODEL_ID,
    "--data_path", TRAIN_DATA_PATH,
    "--output_dir", output_dir,
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
    "--save_total_limit", "2",
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
    "--use_instruction_token", str(QUERY_DATASET_USE_INSTRUCTION_TOKEN),
    "--has_instruction", str(QUERY_DATASET_HAS_INSTRUCTION),
    "--has_hard_negative", str(HAS_HARD_NEGATIVE),
    "--has_modality_hard_negative", str(HAS_MODALITY_HARD_NEGATIVE),
    "--mean_pooling", str(MODEL_MEAN_POOLING),
    "--use_bi_atten", str(MODEL_USE_BI_ATTEN),
    "--use_latent_atten", str(MODEL_USE_LATENT_ATTEN),
    "--use_instruction_mask", str(MODEL_USE_INSTRUCTION_MASK),
    "--use_bi_loss", str(MODEL_USE_BI_LOSS),
    "--use_isotropy_loss", str(MODEL_USE_ISOTROPY_LOSS),
    "--use_cross_entropy_loss", str(MODEL_USE_CROSS_ENTROPY_LOSS),
    "--use_focal_infonce_loss", str(MODEL_USE_FOCAL_INFONCE_LOSS),
    "--use_diht_loss", str(MODEL_USE_DIHT_LOSS),
    "--use_self_attent_pooling", str(MODEL_USE_SELF_ATTENT_POOLING),
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