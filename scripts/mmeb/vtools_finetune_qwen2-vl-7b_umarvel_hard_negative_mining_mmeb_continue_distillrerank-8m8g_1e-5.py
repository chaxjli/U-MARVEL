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

try:
    # ==== 分布式参数 ====
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
    os.environ['NPROC_PER_NODE'] = '8' # 单机单卡 debug 训练时设置为1
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
RUN_ID = f"{MODEL_ID}_umarvel_hard_negative_mining_mmeb_continue_distillrerank-8m8g_1e-5"     # 模型名称
SOURCE_MODEL_NAME = "qwen2-vl-7b_umarvel_hard_negative_mining_mmeb"                           # 负样本的来源
RERANK_MODEL_NAME = "qwen2-vl-7b_umarvel_progressive_transition_mmeb-Rank-Only-Pointwise"     # rerank 模型名称                                     
MODEL_LOCAL_PATH = "./checkpoints/mmeb/qwen2-vl-7b_umarvel_hard_negative_mining_mmeb-merged"  # 基座模型
QUERY_DATA_PATH = "./data/MMEB-train/mmeb_union_query_10percent.jsonl"
CAND_POOL_PATH = "./data/MMEB-train/mmeb_union_candidate.jsonl"
INSTRUCTIONS_PATH = "./data/MMEB-train/data_instruction.json"  # 指令数据集路径
IMAGE_PATH_PREFIX = "./data/MMEB-train"
QUERY_DATASET_HAS_INSTRUCTION = True               # 是否做指令匹配
QUERY_DATASET_USE_INSTRUCTION_TOKEN = True         # 是否使用指令 token
MODEL_MEAN_POOLING = True                          # 是否使用全局平局池化
MODEL_USE_BI_ATTEN = True                          # 是否使用双向注意力模块
MODEL_USE_LATENT_ATTEN = False                     # 是否使用潜在注意力模块
MODEL_USE_INSTRUCTION_MASK = True                  # 是否使用指令 mask
MODEL_USE_RERANK_SCORES = True                     # 是否使用 rerank 分数, 进行蒸馏操作，必须设置为 True

# 蒸馏相关参数
HAS_HARD_NEGATIVE = True                           # 是否使用 local 的融合分数
HAS_MODALITY_HARD_NEGATIVE = False                 # 是否使用 global 的融合分数
MODEL_TOPK_HARD_NEGATIVE = 50                      # 蒸馏 local 时 query 对应的样本数
MODEL_TOPK_MODALITY_HARD_NEGATIVE = 50             # 蒸馏 global 时 query 对应的样本数
HARD_NEGATIVE_PATH = f"./result/score/mmeb/{RERANK_MODEL_NAME}/{SOURCE_MODEL_NAME}/local/fusion_results.jsonl"
MODALITY_HARD_NEGATIVE_PATH = f"./result/score/mmeb/{RERANK_MODEL_NAME}/{SOURCE_MODEL_NAME}/global/fusion_results.jsonl"
QUERY_FEATURE_PATH = f"./result/mmeb/train/{SOURCE_MODEL_NAME}/global/all_query_names2query_features.pth" #  query 特征约束路径
CAND_FEATURE_PATH = f"./result/mmeb/train/{SOURCE_MODEL_NAME}/global/all_cand_ids2cand_features.pth"      #  cand 特征约束路径
MODEL_IGNORE_BATCH_OTHER_SAMPLES = True             # 由于是进行蒸馏，所以需要忽略 batch 内其他负样本
MODEL_USE_FEATURE_CONSTRAINT = False                # 是否使用特征约束


# 下面的参数有且仅有一个为 True
MODEL_USE_CROSS_ENTROPY_LOSS = True                   # 是否使用交叉熵损失
MODEL_USE_FOCAL_INFONCE_LOSS = False                  # 是否使用 FocalInfoNCELoss
MODEL_USE_DIHT_LOSS = False                           # 是否使用 DiHTLoss
MODEL_USE_LLAVE_LOSS = False                          # 是否使用 LLaVELoss

# 其他可以共用的损失函数，不受上述限制
MODEL_USE_KL_CONSTRAINT = True                        # 是否使用 KL 约束
MODEL_USE_JS_CONSTRAINT = False                       # 是否使用 JS 约束
MODEL_USE_GENERALIZED_KL_CONSTRAINT = False           # 是否使用广义 KL 约束
MODEL_USE_MSE_CONSTRAINT = False                      # 是否使用 MSE 约束
MODEL_USE_RANKING_CONSTRAINT = False                  # 是否使用 Ranking 约束

# MODEL_USE_WASSERSTEIN_CONSTRAINT = False             # 是否使用 Wasserstein 约束
# MODEL_USE_F_CONSTRAINT = False                       # 是否使用 F 散度约束

# 定义训练参数
TRAIN_VISION_ENCODER = False
USE_VISION_LORA = False
TRAIN_VISION_PROJECTOR = False
TRAIN_TEMPERATURE = True                    #  是否训练温度参数
USE_LORA = True
Q_LORA = False
LORA_R = 128
LORA_ALPHA = 256
DS_STAGE = "ds_config_zero1"
PER_DEVICE_BATCH_SIZE = 2   # 实验的时候记得修改成 17，保持 120 的特征数量
GRAD_ACCUM = 1
NUM_EPOCHS = 1
LR = 1.0e-5
MODEL_MAX_LEN = 1024

# 创建结果目录
os.makedirs(f"./checkpoints/mmeb/{RUN_ID}", exist_ok=True)
# 创建日志文件位置
log_file_path = f"./checkpoints/mmeb/{RUN_ID}/{TIMENAME}stdout.log"
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
print("负样本的来源: ",SOURCE_MODEL_NAME)
print("负样本的路径: ",HARD_NEGATIVE_PATH)
print("模态负样本的路径: ",MODALITY_HARD_NEGATIVE_PATH)
print("是否忽略 batch 内其他负样本：",MODEL_IGNORE_BATCH_OTHER_SAMPLES)
print("topk hard negative: ",MODEL_TOPK_HARD_NEGATIVE)
print("topk modality hard negative: ",MODEL_TOPK_MODALITY_HARD_NEGATIVE)
print("是否使用 hard negative: ",HAS_HARD_NEGATIVE)
print("是否使用 modality hard negative: ",HAS_MODALITY_HARD_NEGATIVE)
print("是否使用交叉熵损失: ",MODEL_USE_CROSS_ENTROPY_LOSS)
print("是否使用 FocalInfoNCELoss: ",MODEL_USE_FOCAL_INFONCE_LOSS)
print("是否使用 DiHTLoss: ",MODEL_USE_DIHT_LOSS)
print("是否使用 LLaVELoss: ",MODEL_USE_LLAVE_LOSS)
print("是否使用特征约束：",MODEL_USE_FEATURE_CONSTRAINT)
print("是否进行蒸馏: ",MODEL_USE_RERANK_SCORES)
print("是否使用 KL 约束：",MODEL_USE_KL_CONSTRAINT)
print("是否使用 JS 约束：",MODEL_USE_JS_CONSTRAINT)
print("是否使用广义 KL 约束：",MODEL_USE_GENERALIZED_KL_CONSTRAINT)
print("是否使用 MSE 约束：",MODEL_USE_MSE_CONSTRAINT)
print("是否使用 Ranking 约束：",MODEL_USE_RANKING_CONSTRAINT)
print("打印学习率：",LR)

# 构建 torchrun 命令
torchrun_command = [
    "torchrun",
    *DISTRIBUTED_ARGS.split(),
    "train/mmeb/train_mmeb_with_distillrerank.py",
    "--model_id", MODEL_ID,
    "--query_data_path", QUERY_DATA_PATH,
    "--cand_pool_path", CAND_POOL_PATH,
    "--instructions_path", INSTRUCTIONS_PATH,
    "--hard_negative_path", HARD_NEGATIVE_PATH,
    "--modality_hard_negative_path", MODALITY_HARD_NEGATIVE_PATH,
    "--output_dir", f"./checkpoints/mmeb/{RUN_ID}",
    "--report_to", "tensorboard",
    "--run_name", RUN_ID,
    "--deepspeed", f"./ds_configs/{DS_STAGE}.json",
    "--bf16", "True",
    "--num_train_epochs", str(NUM_EPOCHS),
    "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--per_device_eval_batch_size", str(PER_DEVICE_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
    "--eval_strategy", "epoch",
    "--save_strategy", "steps",
    "--save_steps", "600",
    "--save_total_limit", "20",
    "--learning_rate", str(LR),
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "False",
    "--model_max_length", str(MODEL_MAX_LEN),
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",  # 真正做实验的时候记得修改成为 4
    "--train_vision_encoder", str(TRAIN_VISION_ENCODER),
    "--use_vision_lora", str(USE_VISION_LORA),
    "--train_vision_projector", str(TRAIN_VISION_PROJECTOR),
    "--train_temperature", str(TRAIN_TEMPERATURE),
    "--use_lora", str(USE_LORA),
    "--q_lora", str(Q_LORA),
    "--lora_r", str(LORA_R),
    "--lora_alpha", str(LORA_ALPHA),
    "--model_local_path", MODEL_LOCAL_PATH,
    "--image_path_prefix", IMAGE_PATH_PREFIX,
    "--use_instruction_token", str(QUERY_DATASET_USE_INSTRUCTION_TOKEN),
    "--has_instruction", str(QUERY_DATASET_HAS_INSTRUCTION),
    "--has_hard_negative", str(HAS_HARD_NEGATIVE),
    "--has_modality_hard_negative", str(HAS_MODALITY_HARD_NEGATIVE),
    "--mean_pooling", str(MODEL_MEAN_POOLING),
    "--use_bi_atten", str(MODEL_USE_BI_ATTEN),
    "--use_latent_atten", str(MODEL_USE_LATENT_ATTEN),
    "--use_instruction_mask", str(MODEL_USE_INSTRUCTION_MASK),
    "--topk_hard_negative", str(MODEL_TOPK_HARD_NEGATIVE),
    "--topk_modality_hard_negative", str(MODEL_TOPK_MODALITY_HARD_NEGATIVE),
    "--ignore_batch_other_samples", str(MODEL_IGNORE_BATCH_OTHER_SAMPLES),
    "--use_feature_constraint", str(MODEL_USE_FEATURE_CONSTRAINT),
    "--use_rerank_scores", str(MODEL_USE_RERANK_SCORES),
    "--query_feature_path", QUERY_FEATURE_PATH,
    "--cand_feature_path", CAND_FEATURE_PATH,
    "--use_cross_entropy_loss", str(MODEL_USE_CROSS_ENTROPY_LOSS),
    "--use_focal_infonce_loss", str(MODEL_USE_FOCAL_INFONCE_LOSS),
    "--use_diht_loss", str(MODEL_USE_DIHT_LOSS),
    "--use_llave_loss", str(MODEL_USE_LLAVE_LOSS),
    "--use_kl_constraint", str(MODEL_USE_KL_CONSTRAINT),
    "--use_js_constraint", str(MODEL_USE_JS_CONSTRAINT),
    "--use_generalized_kl_constraint", str(MODEL_USE_GENERALIZED_KL_CONSTRAINT),
    "--use_mse_constraint", str(MODEL_USE_MSE_CONSTRAINT),
    "--use_ranking_constraint", str(MODEL_USE_RANKING_CONSTRAINT),

    # "--use_wasserstein_constraint", str(MODEL_USE_WASSERSTEIN_CONSTRAINT),
    # "--use_f_constraint", str(MODEL_USE_F_CONSTRAINT),

]

# 将命令列表转换为字符串
command_str = " ".join(torchrun_command)
# 构建重定向输出到日志文件的命令
redirect_command = f"{command_str} >> {log_file_path}"
assert MODEL_IGNORE_BATCH_OTHER_SAMPLES == True, "模型训练时忽略 batch 内其他负样本的参数设置错误！"
assert MODEL_USE_RERANK_SCORES == True, "模型训练时使用 rerank 分数的参数设置错误！"
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