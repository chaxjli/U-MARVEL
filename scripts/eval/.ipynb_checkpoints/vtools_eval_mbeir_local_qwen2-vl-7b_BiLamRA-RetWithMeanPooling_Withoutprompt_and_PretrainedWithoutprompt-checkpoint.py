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
# 设置分布式环境变量 ----- 提交 vtools 时注销, 单机不需要注销
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

# 评估我们的 model------------------------------- local

MODEL_NAME="qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt"
MODEL_ID = f"./checkpoints/{MODEL_NAME}"
ORIGINAL_MODEL_ID = "./checkpoints/hf_models/Qwen2-VL-7B-Instruct"

# 定义数据文件数组
query_data_paths = [
    "./data/M-BEIR/query/test/mbeir_mscoco_task0_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_mscoco_task3_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_cirr_task7_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_fashioniq_task7_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_webqa_task1_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_nights_task4_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_oven_task6_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_infoseek_task6_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_fashion200k_task0_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_visualnews_task0_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_webqa_task2_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_oven_task8_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_infoseek_task8_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_fashion200k_task3_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_visualnews_task3_test.jsonl",
    "./data/M-BEIR/query/test/mbeir_edis_task2_test.jsonl"
]

query_cand_pool_paths = [
    "./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl"
    # 若所有 query_cand_pool_path 都一样，可简化逻辑
]

cand_pool_paths = [
    "./data/M-BEIR/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_nights_task4_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl",
    "./data/M-BEIR/cand_pool/local/mbeir_edis_task2_cand_pool.jsonl"
]

qrels_paths = [
    "./data/M-BEIR/qrels/test/mbeir_mscoco_task0_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_cirr_task7_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_fashioniq_task7_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_webqa_task1_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_nights_task4_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_oven_task6_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_infoseek_task6_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_fashion200k_task0_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_visualnews_task0_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_webqa_task2_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_oven_task8_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_infoseek_task8_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_fashion200k_task3_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_visualnews_task3_test_qrels.txt",
    "./data/M-BEIR/qrels/test/mbeir_edis_task2_test_qrels.txt"
]

# 动态端口分配
BASE_PORT = 29660


# 获取当前日期和时间
now = datetime.datetime.now()
formatted_with_symbols = now.strftime("%Y-%m-%d %H:%M:%S")
TIMENAME = ''.join(filter(str.isdigit, formatted_with_symbols))

# 设置 umask 并创建输出目录
old_umask = os.umask(0)
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)
os.umask(old_umask)

for i in range(len(query_data_paths)):
    # 提取 qrels 文件的基础名称并去掉 _qrels.txt 后缀
    CURRENT_PORT = BASE_PORT + i  # 当前端口号
    filename = os.path.basename(qrels_paths[i])
    result = filename.replace("_qrels.txt", "")
    # 构建要检查的文件路径
    check_file = f"./result/{MODEL_NAME}_eval_results/{result}_{MODEL_NAME}_candidate_features.pth"

    # 打印 check_file 路径用于调试
    print(f"正在检查文件: {check_file}")

    if os.path.isfile(check_file):
        print(f"文件 {check_file} 存在，跳过本次任务。")
    else:
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--main_process_port={CURRENT_PORT}",
            f"--main_process_ip={MASTER_ADDR}",
            f"--machine_rank={NODE_RANK}",
            f"--num_machines={NNODES}",
            #"--num_processes=4",
            "eval/eval_mbeir_local.py",
            f"--query_data_path={query_data_paths[i]}",
            f"--query_cand_pool_path={query_cand_pool_paths[0]}",
            f"--cand_pool_path={cand_pool_paths[i]}",
            "--instructions_path=./data/M-BEIR/instructions/query_instructions.tsv",
            f"--qrels_path={qrels_paths[i]}",
            f"--save_dir_name=./result/{MODEL_NAME}_eval_results",  # 不需要带 local 因为后续的 global 也会写入这个目录
            f"--original_model_id={ORIGINAL_MODEL_ID}",
            f"--model_id={MODEL_ID}",
            f"--model_id={MODEL_ID}",
            ">",
            f"{OUTPUT}/{MODEL_NAME}_eval_results_local{TIMENAME}stdout.log", # 重定向标准输出（stdout）
            # "2>",
            # f"{OUTPUT}/{MODEL_NAME}_eval_results_local{TIMENAME}stderr.log",
        ]
        subprocess.run(" ".join(command), shell=True)



