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

try:    # 先尝试多机加载环境变量
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量
except: # 多机加载失败，尝试单机加载环境变量
    os.environ['MASTER_ADDR'] = 'localhast'
    os.environ['NNODES'] = '1'
    os.environ['NODE_RANK'] = '0'
    os.environ['NPROC_PER_NODE'] = '8'
    MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
    NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
    NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
    NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量



# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))
# 评估模型
ORIGINAL_MODEL_ID = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"                                  # 原始模型路径
MODEL_NAME = "qwen2-vl-7b_umarvel_distillation-umarvel+"                                            # 模型名称
MODEL_ID = f"./checkpoints/rerank_model/{MODEL_NAME}"                                               # rank 模型路径
# 提供第一阶段检索结果的模型名称
SOURCE_MODEL_NAME = "qwen2-vl-7b_umarvel_distillation"                                              # 第一阶段检索模型名称
os.makedirs(f"./result/result_rank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local", exist_ok=True)
total_log_file = f"./result/result_rank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/{TIMENAME}total_stdout.log"
original_stdout = sys.stdout
sys.stdout = open(total_log_file, 'a', buffering=1)  # 修改这里 buffering=1 表示行缓冲

# 打印模型的参数信息
print("-"*50)
print("MODEL_ID",MODEL_ID)
print("ORIGINAL_MODEL_ID",ORIGINAL_MODEL_ID)
required_vars = ['MASTER_ADDR', 'NNODES', 'NODE_RANK', 'NPROC_PER_NODE']
for var in required_vars:
    if not os.getenv(var):
        print(f"Error: {var} must be set as an environment variable.")
        sys.exit(1)
print(required_vars,[MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])
print([MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])


# 定义所有任务列表
TASK_NAMES = [
    "visualnews_task0",
    "mscoco_task0",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "mscoco_task3",
    "fashion200k_task3",
    "nights_task4",
    "oven_task6", # 6-7h
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8"
]

# 循环执行所有任务-------------------评估 pointwise
for i, TASK_NAME in enumerate(TASK_NAMES):
    # 动态计算端口（起始端口29500）
    CURRENT_PORT = 29500 + i
    # 构建要检查的文件路径 nights_task4_top50_test_queryid2rerank_score.json
    check_file = f"./result/result_rank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/{TASK_NAME}_top50_test_queryid2rerank_score.json"
    print("====================================================================")
    print(f"pointwise ------ Processing task: {TASK_NAME} (Port: {CURRENT_PORT})")
    if os.path.isfile(check_file):
        print(f"文件 {check_file} 存在，跳过本次任务。")
    else:
        # 完整执行命令
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--main_process_port={CURRENT_PORT}",
            f"--main_process_ip={MASTER_ADDR}",
            f"--machine_rank={NODE_RANK}",
            f"--num_machines={NNODES}",
            f"--num_processes={NPROC_PER_NODE * NNODES}",
            "eval/eval_mbeir_rerank_pointwise.py",
            f"--query_data_path=./data/M-BEIR/query/test/mbeir_{TASK_NAME}_test.jsonl",
            "--cand_pool_path=./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl",
            "--instructions_path=./data/M-BEIR/instructions/query_instructions.tsv",
            f"--model_id={MODEL_ID}",
            f"--original_model_id={ORIGINAL_MODEL_ID}",
            f"--ret_query_data_path=./result/{SOURCE_MODEL_NAME}_eval_results_finetune/local/mbeir_{TASK_NAME}_test_{SOURCE_MODEL_NAME}_query_names.json",
            f"--ret_cand_data_path=./result/{SOURCE_MODEL_NAME}_eval_results_finetune/local/mbeir_{TASK_NAME}_test_{SOURCE_MODEL_NAME}_cand_names.json",
            f"--save_dir_name=./result/result_rank/{MODEL_NAME}/{SOURCE_MODEL_NAME}/local/",
            "--rank_num=50",
            f"--save_name={TASK_NAME}_top50"
        ]
        try:
            # 打开总的日志文件以追加模式写入
            sys.stdout.flush()  # 强制刷新缓冲区
            with open(total_log_file, 'a') as log_file:
                # 执行命令并将标准输出和标准错误输出重定向到日志文件
                print(f"正在执行命令: {' '.join(command)}")
                print(f"任务开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"任务名称: {TASK_NAME}")
                # subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
                # 执行命令并将标准输出重定向到日志文件，标准错误输出到终端
                subprocess.run(command, stdout=log_file, stderr=None, check=True)
            print(f"任务执行完成，日志已记录到 {total_log_file}")
            print("*" * 50)
        except subprocess.CalledProcessError as e:
            print(f"任务执行失败: {e}")
# 恢复标准输出
sys.stdout.close()
sys.stdout = original_stdout