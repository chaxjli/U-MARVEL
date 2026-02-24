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
    os.environ['NPROC_PER_NODE'] = '1'
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
MODEL_NAME="qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature_continue_distillrerank-8m8g_1e-5"
MODEL_ID = f"./checkpoints/{MODEL_NAME}"
ORIGINAL_MODEL_ID ="./checkpoints/qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature-merged"
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
os.makedirs(f"./result/{MODEL_NAME}_eval_results_finetune/mmeb_multi_gpu_eval", exist_ok=True)
total_log_file = f"./result/{MODEL_NAME}_eval_results_finetune/mmeb_multi_gpu_eval/{TIMENAME}total_stdout.log"
original_stdout = sys.stdout
sys.stdout = open(total_log_file, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印模型的参数信息
print("-"*50)
print("MODEL_NAME",MODEL_NAME)
print("MODEL_ID",MODEL_ID)
print(required_vars,[MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])

CURRENT_PORT = 29671
command = [
    "accelerate",
    "launch",
    # "--multi_gpu",
    f"--main_process_port={CURRENT_PORT}",
    f"--main_process_ip={MASTER_ADDR}",
    f"--machine_rank={NODE_RANK}",
    f"--num_machines={NNODES}",
    f"--num_processes={NPROC_PER_NODE * NNODES}",
    "eval/eval_mmeb_finetune_multi_gpu.py",
    f"--save_dir_name=./result/{MODEL_NAME}_eval_results_finetune",
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
    sys.stdout.flush()  # 强制刷新缓冲区
    with open(total_log_file, 'a') as log_file:
        # 执行命令并将标准输出和标准错误输出重定向到日志文件
        # subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        # 执行命令并将标准输出重定向到日志文件，标准错误输出到终端
        subprocess.run(command, stdout=log_file, stderr=None, check=True)
    print(f"任务执行完成，日志已记录到 {total_log_file}")
except subprocess.CalledProcessError as e:
    print(f"任务执行失败: {e}")


# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout