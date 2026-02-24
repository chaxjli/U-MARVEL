# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct \
#     --model_id the_model_path_after_the_first_stage_of_pre-training \
#     --save_path the_path_you_want_to_save

source /home/user_name/miniconda3/bin/activate && conda activate u-marvel_qwen3

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel_qwen3/bin/:$PATH"

# CUDA_VISIBLE_DEVICES='3' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge_multi_gpu_qwen3_vl.py \
#     --original_model_id ./checkpoints/qwen3-vl-4b_m-beir_stage2_model-merged \
#     --model_id ./checkpoints/qwen3-vl-4b_m-beir_stage3_model \
#     --save_path ./checkpoints/qwen3-vl-4b_m-beir_stage3_model-merged \

CUDA_VISIBLE_DEVICES='0' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge_multi_gpu_qwen3_vl.py \
    --original_model_id ./checkpoints/qwen3-vl-8b_nli_pertrain_stage1_model_casual_atten-merged \
    --model_id ./checkpoints/qwen3-vl-8b_cc3m_llm_stage1_model_casual_atten_lr1e-4_bsz60_epoch1_lora128_alpha256_2m8g \
    --save_path ./checkpoints/qwen3-vl-8b_cc3m_llm_stage1_model_casual_atten_lr1e-4_bsz60_epoch1_lora128_alpha256_2m8g-merged \




  