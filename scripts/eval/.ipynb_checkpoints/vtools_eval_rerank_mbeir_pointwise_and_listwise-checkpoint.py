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
# print(os.environ)
# # 设置分布式环境变量 ----- 提交 vtools 时注销
os.environ['MASTER_ADDR'] = 'localhast'
os.environ['NNODES'] = '1'
os.environ['NODE_RANK'] = '0'
os.environ['NPROC_PER_NODE'] = '8'


# ==== 分布式参数 ====
MASTER_ADDR = os.environ.get('MASTER_ADDR')             # master_ip
NNODES = int(os.environ.get('NNODES'))                  # workers 总节点数
NODE_RANK = int(os.environ.get('NODE_RANK'))            # worker_id 当前机器节点 Rank (主节点为 0)
NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE'))  # gpu_num 每个节点的 NPU 数量

# 检查分布式参数是否设置
required_vars = ['MASTER_ADDR', 'NNODES', 'NODE_RANK', 'NPROC_PER_NODE']
print([MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE])

for var in required_vars:
    if not os.getenv(var):
        print(f"Error: {var} must be set as an environment variable.")
        sys.exit(1)

# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 设置 umask 并创建输出目录
old_umask = os.umask(0)
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)
os.umask(old_umask)


# 评估模型
ORIGINAL_MODEL_ID = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID = "./checkpoints/qwen2-vl-7b_LamRA-Rank"

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
    "oven_task6",
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
    check_file = f"./result/mbeir_rerank_files/{TASK_NAME}_top50_test_queryid2rerank_score.json"
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
            "eval/eval_mbeir_rerank_pointwise.py",
            f"--query_data_path=./data/M-BEIR/query/test/mbeir_{TASK_NAME}_test.jsonl",
            "--cand_pool_path=./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl",
            "--instructions_path=./data/M-BEIR/instructions/query_instructions.tsv",
            f"--model_id={MODEL_ID}",
            f"--original_model_id={ORIGINAL_MODEL_ID}",
            f"--ret_query_data_path=./LamRA_Ret_eval_results/mbeir_{TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_query_names.json",
            f"--ret_cand_data_path=./LamRA_Ret_eval_results/mbeir_{TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json",
            "--save_dir_name=./result/mbeir_rerank_files",
            "--rank_num=50",
            f"--save_name={TASK_NAME}_top50"
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed: {TASK_NAME}")
        except subprocess.CalledProcessError as e:
            print(f"执行任务 {TASK_NAME} 时出错: {e}")
        print("====================================================================")

# 循环执行所有任务-------------------评估 listwise
for i, TASK_NAME in enumerate(TASK_NAMES):
    # 动态计算端口（起始端口29600）
    CURRENT_PORT = 29600 + i
    # 构建要检查的文件路径 mscoco_task3_top5_all_test_queryid2rerank_outputs_listwise.json
    check_file = f"./result/mbeir_rerank_files/{TASK_NAME}_top5_all_test_queryid2rerank_outputs_listwise.json"
    print("====================================================================")
    print(f"listwise ------ Processing task: {TASK_NAME} (Port: {CURRENT_PORT})")
    if os.path.isfile(check_file):
        print(f"文件 {check_file} 存在，跳过本次任务。")
    else:
        # 完整执行命令
        # 如果显示 NPU 显存过大失败使用 eval_mbeir_rerank_listwise_comment.py 文件
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--main_process_port={CURRENT_PORT}",
            "eval/eval_mbeir_rerank_listwise_comment.py",
            f"--query_data_path=./data/M-BEIR/query/test/mbeir_{TASK_NAME}_test.jsonl",
            "--cand_pool_path=./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl",
            "--instructions_path=./data/M-BEIR/instructions/query_instructions.tsv",
            f"--model_id={MODEL_ID}",
            f"--original_model_id={ORIGINAL_MODEL_ID}",
            f"--ret_query_data_path=./LamRA_Ret_eval_results/mbeir_{TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_query_names.json",
            f"--ret_cand_data_path=./LamRA_Ret_eval_results/mbeir_{TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json",
            "--rank_num=5",
            f"--save_name={TASK_NAME}_top5_all",
            "--save_dir_name=./result/mbeir_rerank_files",
            "--image_path_prefix=./data/M-BEIR",
            "--batch_size=4"
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed: {TASK_NAME}")
        except subprocess.CalledProcessError as e:
            print(f"执行任务 {TASK_NAME} 时出错: {e}")
        print("====================================================================")

# # Get the reranking results on M-BEIR
# try:
#     subprocess.run(["sh", "./scripts/eval/get_rerank_results_mbeir.sh"], check=True)
# except subprocess.CalledProcessError as e:
#     print(f"执行脚本 get_rerank_results_mbeir.sh 时出错: {e}")


