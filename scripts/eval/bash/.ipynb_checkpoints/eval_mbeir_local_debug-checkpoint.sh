source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

# MODEL_ID="./checkpoints/qwen2-vl-7b_LamRA-Ret"
# ORIGINAL_MODEL_ID=./checkpoints/hf_models/Qwen2-VL-7B-Instruct

# eval_mbeir.py # 评估自己的模型
# eval_mbeir_author_model.py # 评估作者的模型

MODEL_ID="./author_model/LamRA-Ret"
ORIGINAL_MODEL_ID=./checkpoints/hf_models/Qwen2-VL-7B-Instruct

# 定义数据文件数组
query_data_paths=(
    "./data/M-BEIR/query/test/mbeir_mscoco_task0_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_mscoco_task3_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_cirr_task7_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_fashioniq_task7_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_webqa_task1_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_nights_task4_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_oven_task6_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_infoseek_task6_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_fashion200k_task0_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_visualnews_task0_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_webqa_task2_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_oven_task8_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_infoseek_task8_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_fashion200k_task3_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_visualnews_task3_test.jsonl"
    "./data/M-BEIR/query/test/mbeir_edis_task2_test.jsonl"
)

query_cand_pool_paths=(
    "./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl"
    # 若所有 query_cand_pool_path 都一样，可简化逻辑
)

cand_pool_paths=(
    "./data/M-BEIR/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_nights_task4_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl"
    "./data/M-BEIR/cand_pool/local/mbeir_edis_task2_cand_pool.jsonl"
)

qrels_paths=(
    "./data/M-BEIR/qrels/test/mbeir_mscoco_task0_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_cirr_task7_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_fashioniq_task7_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_webqa_task1_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_nights_task4_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_oven_task6_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_infoseek_task6_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_fashion200k_task0_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_visualnews_task0_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_webqa_task2_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_oven_task8_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_infoseek_task8_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_fashion200k_task3_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_visualnews_task3_test_qrels.txt"
    "./data/M-BEIR/qrels/test/mbeir_edis_task2_test_qrels.txt"
)

# 定义端口数组
ports=(
    29509
)
# 动态端口分配
BASE_PORT=29500


for i in "${!query_data_paths[@]}"; do
    # 提取 qrels 文件的基础名称并去掉 _qrels.txt 后缀
    CURRENT_PORT=$((BASE_PORT + i)) # 当前端口号
    filename=$(basename "${qrels_paths[$i]}")
    result=$(echo "$filename" | sed 's/_qrels.txt//')
    # 构建要检查的文件路径
    check_file="./LamRA_Ret_eval_results/${result}_qwen2-vl-7b_LamRA-Ret_candidate_features.pth"

    # 打印 check_file 路径用于调试
    echo "正在检查文件: $check_file"

    if [ -f "$check_file" ]; then
        echo "文件 $check_file 存在，跳过本次任务。"
    else
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port "$CURRENT_PORT" eval/eval_mbeir.py \
            --query_data_path "${query_data_paths[$i]}" \
            --query_cand_pool_path "${query_cand_pool_paths[0]}" \
            --cand_pool_path "${cand_pool_paths[$i]}" \
            --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
            --qrels_path "${qrels_paths[$i]}" \
            --original_model_id "$ORIGINAL_MODEL_ID" \
            --model_id "$MODEL_ID"
    fi
done
