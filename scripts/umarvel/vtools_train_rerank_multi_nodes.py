import os
import sys
import pathlib

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
# print(os.environ)




# 设置 LD_LIBRARY_PATH 环境变量
# ld_library_path = "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:" + os.getenv('LD_LIBRARY_PATH', '')
# os.environ['LD_LIBRARY_PATH'] = ld_library_path

# 设置 PYTHONPATH 环境变量
# python_path = "/usr/local/Ascend/pyACL/python/site-packages:" + os.getenv('PYTHONPATH', '')
# os.environ['PYTHONPATH'] = python_path

# # 设置分布式环境变量 ----- 提交 vtools 时注销
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['NNODES'] = '1'
# os.environ['NODE_RANK'] = '0'
# os.environ['NPROC_PER_NODE'] = '8'


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

# ==== 模型训练参数 (根据需求修改) ====
MODEL_ID = "qwen2-vl-7b"
MODEL_LOCAL_PATH = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
QUERY_DATA_PATH = "./data/M-BEIR/query/union_train/mbeir_union_up_train.jsonl"
CAND_POOL_PATH = "./data/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl"
INSTRUCTIONS_PATH = "./data/M-BEIR/instructions/query_instructions.tsv"
RERANK_DATA_PATH = "./data/rerank_data_for_training/rerank_data_all.json"
IMAGE_PATH_PREFIX = "./data/M-BEIR"

TRAIN_VISION_ENCODER = False
USE_VISION_LORA = False
TRAIN_VISION_PROJECTOR = False

USE_LORA = True
Q_LORA = False
LORA_R = 128
LORA_ALPHA = 256

RUN_ID = f"{MODEL_ID}_LamRA-Rank"

DS_STAGE = "ds_config_zero1"
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 1

LR = 2e-5
MODEL_MAX_LEN = 2048

# 设置 umask 并创建输出目录
old_umask = os.umask(0)
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)
os.umask(old_umask)

# ==== 启动训练 ====
train_command = f"torchrun {DISTRIBUTED_ARGS} train/train_rerank_multi_nodes.py " \
                f"--model_id {MODEL_ID} " \
                f"--query_data_path {QUERY_DATA_PATH} " \
                f"--cand_pool_path {CAND_POOL_PATH} " \
                f"--instructions_path {INSTRUCTIONS_PATH} " \
                f"--output_dir ./checkpoints/{RUN_ID} " \
                f"--report_to tensorboard " \
                f"--run_name {RUN_ID} " \
                f"--deepspeed ./ds_configs/{DS_STAGE}.json " \
                f"--bf16 True " \
                f"--num_train_epochs {NUM_EPOCHS} " \
                f"--per_device_train_batch_size {PER_DEVICE_BATCH_SIZE} " \
                f"--per_device_eval_batch_size {PER_DEVICE_BATCH_SIZE} " \
                f"--gradient_accumulation_steps {GRAD_ACCUM} " \
                f"--eval_strategy \"no\" " \
                f"--save_strategy \"steps\" " \
                f"--save_steps 2000 " \
                f"--save_total_limit 20 " \
                f"--learning_rate {LR} " \
                f"--weight_decay 0. " \
                f"--warmup_ratio 0.03 " \
                f"--lr_scheduler_type \"cosine\" " \
                f"--logging_steps 1 " \
                f"--tf32 False " \
                f"--model_max_length {MODEL_MAX_LEN} " \
                f"--gradient_checkpointing True " \
                f"--dataloader_num_workers 4 " \
                f"--train_vision_encoder {TRAIN_VISION_ENCODER} " \
                f"--use_vision_lora {USE_VISION_LORA} " \
                f"--train_vision_projector {TRAIN_VISION_PROJECTOR} " \
                f"--use_lora {USE_LORA} " \
                f"--q_lora {Q_LORA} " \
                f"--lora_r {LORA_R} " \
                f"--lora_alpha {LORA_ALPHA} " \
                f"--model_local_path {MODEL_LOCAL_PATH} " \
                f"--rerank_data_path {RERANK_DATA_PATH} " \
                f"--image_path_prefix {IMAGE_PATH_PREFIX} > {OUTPUT}/log.log"

try:
    result = os.system(train_command)
    if result != 0:
        raise Exception("torchrun command failed.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)





