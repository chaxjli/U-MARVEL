import os
import sys
import pathlib
import subprocess
import datetime

# ==== NPU 环境配置 ====
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
os.environ["PATH"] = f'''/users/miniconda3/envs/ U-MARVEL_npu/bin/:{ori_env_path}'''

# 设置分布式环境变量
try:    # 先尝试多机加载环境变量
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量
except: # 多机加载失败，尝试单机加载环境变量
    print("多机加载环境变量失败，尝试单机加载环境变量")
    os.environ['MASTER_ADDR'] = 'localhast'
    os.environ['NNODES'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['NPROC_PER_NODE'] = '8'
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量

# 检查分布式参数是否设置
required_vars = ['MASTER_ADDR', 'NNODES', 'NODE_RANK', 'NPROC_PER_NODE']
for var in required_vars:
    if not os.getenv(var):
        print(f"Error: {var} must be set as an environment variable.")
        sys.exit(1)

# 模型评估参数   -----  请根据实际情况修改
MODEL_NAME="U-MARVEL-Qwen2VL-7B-Instruct"
MODEL_ID = f"./checkpoints/{MODEL_NAME}"
ORIGINAL_MODEL_ID =f"./checkpoints/{MODEL_NAME}"
QUERY_DATASET_HAS_INSTRUCTION=True              # 是否做指令匹配
QUERY_DATASET_USE_INSTRUCTION_TOKEN=True        # 是否使用指令 token
MODEL_MEAN_POOLING=True                         # 是否使用全局平局池化
MODEL_USE_BI_ATTEN=True                         # 是否使用双向注意力模块
MODEL_USE_LATENT_ATTEN=False                    # 是否使用潜在注意力模块
MODEL_USE_INSTRUCTION_MASK=True                 # 是否使用指令 mask

# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 创建结果目录
os.makedirs(f"./result/{MODEL_NAME}_eval_results_finetune/local", exist_ok=True)
total_log_file = f"./result/{MODEL_NAME}_eval_results_finetune/local/{TIMENAME}total_stdout.log"
original_stdout = sys.stdout
sys.stdout = open(total_log_file, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印模型的参数信息
print("-"*50)
print("MODEL_NAME",MODEL_NAME)
print("MODEL_ID",MODEL_ID)
print(required_vars,[MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])

CURRENT_PORT = 29670
command = [
    "accelerate",
    "launch",
    "--multi_gpu",
    f"--main_process_port={CURRENT_PORT}",
    f"--main_process_ip={MASTER_ADDR}",
    f"--machine_rank={NODE_RANK}",
    f"--num_machines={NNODES}",
    f"--num_processes={NPROC_PER_NODE * NNODES}",
    "eval/eval_mbeir_local_finetune.py",
    "--instructions_path=./data/M-BEIR/instructions/query_instructions.tsv",
    f"--save_dir_name=./result/{MODEL_NAME}_eval_results_finetune/local",
    f"--original_model_id={ORIGINAL_MODEL_ID}",
    f"--model_id={MODEL_ID}",
    f"--query_dataset_has_instruction={QUERY_DATASET_HAS_INSTRUCTION}",
    f"--query_dataset_use_instruction_token={QUERY_DATASET_USE_INSTRUCTION_TOKEN}",
    f"--model_mean_pooling={MODEL_MEAN_POOLING}",
    f"--model_use_bi_atten={MODEL_USE_BI_ATTEN}",
    f"--model_use_latent_atten={MODEL_USE_LATENT_ATTEN}",
    f"--model_use_instruction_mask={MODEL_USE_INSTRUCTION_MASK}",
]
try:
    # 打开总的日志文件以追加模式写入
    sys.stdout.flush()
    with open(total_log_file, 'a') as log_file:
        subprocess.run(command, stdout=log_file, stderr=None, check=True)
    print(f"任务执行完成，日志已记录到 {total_log_file}")
except subprocess.CalledProcessError as e:
    print(f"任务执行失败: {e}")


# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout