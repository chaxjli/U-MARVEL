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


# 选取 zero-shot 评估的模型，开源的 huggingface 模型 或者 本地 epoch 1 的大模型, "listwise" or "pointwise"
MODEL_NAME = "Qwen2-VL-7B-Instruct"
ORIGINAL_MODEL_ID = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID = f"./checkpoints/hf_models/{MODEL_NAME}"

SOURCE_MODEL_NAME = "qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_focal_infonce_loss_8m8g_4xlr" # zero-shot 的数据来源
SOURCE_MODEL_ID = f"./checkpoints/hf_models/{SOURCE_MODEL_NAME}"

# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 创建结果目录
os.makedirs(f"./result/result_zero_shot_rerank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local", exist_ok=True)
total_log_file = f"./result/result_zero_shot_rerank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/{TIMENAME}total_stdout.log"
original_stdout = sys.stdout
sys.stdout = open(total_log_file, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印模型的参数信息
print("-"*50)
print("MODEL_NAME",MODEL_NAME)
print("MODEL_ID",MODEL_ID)
print("SOURCE_MODEL_NAME",SOURCE_MODEL_NAME)
print("SOURCE_MODEL_ID",SOURCE_MODEL_ID)
print("ORIGINAL_MODEL_ID",ORIGINAL_MODEL_ID)
print(required_vars,[MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])


# 定义任务列表
tasks = [
    "mscoco_task3",
    "mscoco_task0",
    "visualnews_task0",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "fashion200k_task3",
    "nights_task4",
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8",
    "oven_task6",
    ]

for TASK_NAME in tasks:
    # 检查该任务文件是否存在，如果存在则跳过
    check_file = f"./result/result_zero_shot_rerank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/{TASK_NAME}_top50_test_queryid2rerank_score.json"
    if os.path.exists(check_file):
        print(f"任务 {TASK_NAME} 的结果文件已存在，跳过该任务。")
        continue
    # 如果不存在，则执行任务
    # f"--machine_rank={NODE_RANK}",
    # f"--num_machines={NNODES}",
    # f"--num_processes={NPROC_PER_NODE * NNODES}",
    print(f"开始执行任务 {TASK_NAME} 的评估")
    command = [
        "accelerate", "launch", "--multi_gpu", 
        "--main_process_port", "29508",
        "--main_process_ip",MASTER_ADDR,
        "--num_machines",str(NNODES),
        "--machine_rank",str(NODE_RANK),
        "--num_processes", str(NPROC_PER_NODE * NNODES),
        "eval/eval_mbeir_zero_shot_rerank_pointwise.py",
        "--query_data_path", f"./data/M-BEIR/query/test/mbeir_{TASK_NAME}_test.jsonl",
        "--cand_pool_path", "./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl",
        "--instructions_path", "./data/M-BEIR/instructions/query_instructions.tsv",
        "--model_id", MODEL_ID,
        "--original_model_id", ORIGINAL_MODEL_ID,
        "--ret_query_data_path", f"./result/{SOURCE_MODEL_NAME}_eval_results_finetune/local/mbeir_{TASK_NAME}_test_{SOURCE_MODEL_NAME}_query_names.json",
        "--ret_cand_data_path", f"./result/{SOURCE_MODEL_NAME}_eval_results_finetune/local/mbeir_{TASK_NAME}_test_{SOURCE_MODEL_NAME}_cand_names.json",
        "--rank_num", "50",
        "--save_name", f"{TASK_NAME}_top50",
        "--save_dir_name", f"./result/result_zero_shot_rerank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/",
    ]

    try:
        # 打开总的日志文件以追加模式写入
        sys.stdout.flush()  # 强制刷新缓冲区
        with open(total_log_file, 'a') as log_file:
            print(f"正在执行命令: {' '.join(command)}")
            print(f"任务开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"任务名称: {TASK_NAME}")
            # 执行命令并将标准输出和标准错误输出重定向到日志文件
            # subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
            # 执行命令并将标准输出重定向到日志文件，标准错误输出到终端
            subprocess.run(command, stdout=log_file, stderr=None, check=True)
        print(f"任务执行完成，日志已记录到 {total_log_file}")
    except subprocess.CalledProcessError as e:
        print(f"任务执行失败: {e}")

print("所有任务执行完成，日志已记录到", total_log_file)
# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout