# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct \
#     --model_id the_model_path_after_the_first_stage_of_pre-training \
#     --save_path the_path_you_want_to_save

source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

# --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
# CUDA_VISIBLE_DEVICES='0' accelerate launch --num_processes=1 --main_process_port 29500 merge_lora/merge_multi_gpu.py \
#     --original_model_id ./checkpoints/BiLamRA_Ret_PretrainWithMeanpooling_Withoutprompt_token_cls-merged \
#     --model_id ./checkpoints/qwen2-vl-7b_cc3m_sharegpt4v_llm \
#     --save_path ./checkpoints/qwen2-vl-7b_cc3m_sharegpt4v_llm-merged \

# CUDA_VISIBLE_DEVICES='1' accelerate launch --num_processes=1 --main_process_port 29501 merge_lora/merge_multi_gpu.py \
#     --original_model_id ./checkpoints/BiLamRA_Ret_PretrainWithMeanpooling_Withoutprompt_token_cls-merged \
#     --model_id ./checkpoints/qwen2-vl-7b_cc3m_sharegpt4v_llm_cvprojector \
#     --save_path ./checkpoints/qwen2-vl-7b_cc3m_sharegpt4v_llm_cvprojector-merged \

CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge_multi_gpu.py \
    --original_model_id ./checkpoints/held-out/qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_held-out-merged \
    --model_id ./checkpoints/held-out/qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature_held-out \
    --save_path ./checkpoints/held-out/qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_8m8g_4xlr_train_temperature_global_mixed_hard_neg_all_8m8g_1point5e-5-two_strategy_train_temperature_held-out-merged \

# accelerate launch --multi_gpu --main_process_port 29500 merge_lora/merge_multi_gpu.py \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2-vl-7b_LamRA_Ret_Pretrain \
#     --save_path ./checkpoints/LamRA-Ret-Pretrained-merged_single_gpu

# ASCEND_VISIBLE_DEVICES='0' accelerate launch  --main_process_port 29500 merge_lora/merge_multi_gpu.py \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2-vl-7b_LamRA_Ret_Pretrain \
#     --save_path ./checkpoints/LamRA-Ret-Pretrained-merged_multi_gpu

# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 0 merge_lora/merge.py \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2-vl-7b_LamRA_Ret_Pretrain \
#     --save_path ./checkpoints/LamRA-Ret-Pretrained-merged



# accelerate launch --multi_gpu --main_process_port 0 merge_lora/merge.py \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2-vl-7b_LamRA_Ret_Pretrain \
#     --save_path ./checkpoints/LamRA-Ret-Pretrained-merged




  