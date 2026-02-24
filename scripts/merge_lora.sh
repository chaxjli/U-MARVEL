# 
source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge.py \
    --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct/ \  # Lora 前的基座模型路径
    --model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_nli \
    --save_path ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_nli_-merged \

# CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge.py \
#     --original_model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_nli_-merged \  # Lora 前的基座模型路径
#     --model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_cc3m \
#     --save_path ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_cc3m-merged \

# CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge.py \
#     --original_model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_cc3m_-merged \  # Lora 前的基座模型路径
#     --model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_m-beir \
#     --save_path ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_m-beir-merged \

# CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge.py \
#     --original_model_id ./checkpoints/qwen2-vl-7b_umarvel_progressive_transition_m-beir-merged \  # Lora 前的基座模型路径
#     --model_id ./checkpoints/qwen2-vl-7b_umarvel_hard_negative_mining \
#     --save_path ./checkpoints/qwen2-vl-7b_umarvel_hard_negative_mining-merged \

# CUDA_VISIBLE_DEVICES='2' accelerate launch --num_processes=1 --main_process_port 29502 merge_lora/merge.py \
#     --original_model_id qwen2-vl-7b_umarvel_hard_negative_mining-merged \  # Lora 前的基座模型路径
#     --model_id ./checkpoints/qwen2-vl-7b_umarvel_distillation \
#     --save_path ./checkpoints/qwen2-vl-7b_umarvel_distillation-merged \





  