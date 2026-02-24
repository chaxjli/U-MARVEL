source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

# 评估作者模型
ORIGINAL_MODEL_ID="./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID="./checkpoints/qwen2-vl-7b_LamRA-Rank"

# 定义所有任务列表
TASK_NAMES=(
    visualnews_task0
    mscoco_task0
    fashion200k_task0
    webqa_task1
    edis_task2
    webqa_task2
    visualnews_task3
    mscoco_task3
    fashion200k_task3
    nights_task4
    oven_task6
    infoseek_task6
    fashioniq_task7
    cirr_task7
    oven_task8
    infoseek_task8
)



# 循环执行所有任务-------------------评估 pointwise
for i in "${!TASK_NAMES[@]}"; do
    # 动态计算端口（起始端口29500）
    CURRENT_PORT=$((29500 + i))
    # 动态生成路径
    TASK_NAME="${TASK_NAMES[$i]}" 
    # 构建要检查的文件路径 nights_task4_top50_test_queryid2rerank_score.json
    check_file="./result/mbeir_rerank_files/${TASK_NAME}_top50_test_queryid2rerank_score.json"
    echo "===================================================================="
    echo "pointwise ------ Processing task: $TASK_NAME (Port: $CURRENT_PORT)"
    if [ -f "$check_file" ]; then
        echo "文件 $check_file 存在，跳过本次任务。"
    else
        # 完整执行命令
        accelerate launch --multi_gpu --main_process_port $CURRENT_PORT \
            eval/eval_mbeir_rerank_pointwise.py \
            --query_data_path "./data/M-BEIR/query/test/mbeir_${TASK_NAME}_test.jsonl" \
            --cand_pool_path "./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl" \
            --instructions_path "./data/M-BEIR/instructions/query_instructions.tsv" \
            --model_id "$MODEL_ID" \
            --original_model_id "$ORIGINAL_MODEL_ID" \
            --ret_query_data_path "./LamRA_Ret_eval_results/mbeir_${TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_query_names.json" \
            --ret_cand_data_path "./LamRA_Ret_eval_results/mbeir_${TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json" \
            --save_dir_name "./result/mbeir_rerank_files" \
            --rank_num 50 \
            --save_name "${TASK_NAME}_top50"        
        echo "Successfully processed: $TASK_NAME"
        echo "===================================================================="
    fi
done

# 循环执行所有任务-------------------评估 listwise
for i in "${!TASK_NAMES[@]}"; do
    # 动态计算端口（起始端口29500）
    CURRENT_PORT=$((29600 + i))
    # 动态生成路径
    TASK_NAME="${TASK_NAMES[$i]}"
    # 构建要检查的文件路径 mscoco_task3_top5_all_test_queryid2rerank_outputs_listwise.json
    check_file="./result/mbeir_rerank_files/${TASK_NAME}_top5_all_test_queryid2rerank_outputs_listwise.json"
    
    echo "===================================================================="
    echo "listwise ------ Processing task: $TASK_NAME (Port: $CURRENT_PORT)"
    if [ -f "$check_file" ]; then
        echo "文件 $check_file 存在，跳过本次任务。"
    else
        # 完整执行命令
        # 如果显示 NPU 显存过大失败使用 eval_mbeir_rerank_listwise_comment.py 文件
        accelerate launch --multi_gpu --main_process_port $CURRENT_PORT \
            eval/eval_mbeir_rerank_listwise_comment.py \
            --query_data_path ./data/M-BEIR/query/test/mbeir_${TASK_NAME}_test.jsonl \
            --cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
            --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
            --model_id $MODEL_ID \
            --original_model_id $ORIGINAL_MODEL_ID \
            --ret_query_data_path ./LamRA_Ret_eval_results/mbeir_${TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_query_names.json \
            --ret_cand_data_path ./LamRA_Ret_eval_results/mbeir_${TASK_NAME}_test_qwen2-vl-7b_LamRA-Ret_cand_names.json \
            --rank_num 5 \
            --save_name ${TASK_NAME}_top5_all \
            --save_dir_name ./result/mbeir_rerank_files \
            --image_path_prefix ./data/M-BEIR \
            --batch_size 4        
        echo "Successfully processed: $TASK_NAME"
        echo "===================================================================="
    fi
done

# Get the reranking results on M-BEIR
sh ./scripts/eval/get_rerank_results_mbeir.sh







