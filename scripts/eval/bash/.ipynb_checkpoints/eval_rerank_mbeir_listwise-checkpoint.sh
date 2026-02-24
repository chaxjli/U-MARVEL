source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

# 评估作者模型
ORIGINAL_MODEL_ID="./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID="./author_model/LamRA-Rank"

TASK_NAME=mscoco_task3
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29520 eval/eval_mbeir_rerank_listwise.py \
    --query_data_path ./data/M-BEIR/query/test/mbeir_${TASK_NAME}_test.jsonl \
    --cand_pool_path ./data/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --instructions_path ./data/M-BEIR/instructions/query_instructions.tsv \
    --model_id $MODEL_ID \
    --original_model_id $ORIGINAL_MODEL_ID \
    --ret_query_data_path ./LamRA_Ret_eval_results_author_model/mbeir_${TASK_NAME}_test_LamRA-Ret_query_names.json \
    --ret_cand_data_path ./LamRA_Ret_eval_results_author_model/mbeir_${TASK_NAME}_test_LamRA-Ret_cand_names.json \
    --rank_num 5 \
    --save_name ${TASK_NAME}_top5_all \
    --save_dir_name ./result/mbeir_rerank_files_author_model \
    --image_path_prefix ./data/M-BEIR \
    --batch_size 4