
source /home/user_name/miniconda3/bin/activate && conda activate u-marvel_qwen3
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel_qwen3/bin/:$PATH"
# export ASCEND_LAUNCH_BLOCKING=1 # debug 时候开启，正常运行时候关闭

# MODEL_NAME="qwen2_5-vl-7b_nli_pertrain_stage1_model_lr2e-4_bsz120_epoch2_lorar32_alpha64"
# MODEL_NAME="qwen2_5-vl-7b_nli_pertrain_stage1_model_casual_atten"
MODEL_NAME="qwen2_5-vl-7b_nli_pertrain_stage1_model_casual_atten_lr4e-4_bsz72_epoch2_lora64_alpha128"
mkdir -p "./result/pretrain/${MODEL_NAME}/flickr/"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29500 eval/eval_zeroshot/eval_flickr.py \
CUDA_VISIBLE_DEVICES='0,1' accelerate launch --multi_gpu --main_process_port 29500 eval/eval_zeroshot/eval_flickr.py \
--image_data_path ./data/flickr/images \
--text_data_path ./data/flickr/flickr_text.json \
--original_model_id ./checkpoints/hf_models/Qwen2_5-VL-7B-Instruct \
--save_dir_name "./result/pretrain/${MODEL_NAME}/flickr/" \
--model_id ./checkpoints/${MODEL_NAME} \
--use_instruction_mask "True" \
--use_bi_atten "False" \
--mean_pooling "False" \
>> ./result/pretrain/${MODEL_NAME}/flickr/pretrain_flickr.log


# 我要测试的模型列表：
# qwen2_5-vl-7b_cc3m_llm_stage1_model
# qwen2_5-vl-7b_cc3m_llm_stage1_model_lr2e-4_bsz60_epoch1_lorar32_alpha64_8m8g
# qwen2_5-vl-7b_cc3m_llm_stage1_model_lr4e-4_bsz60_epoch1_lorar32_alpha64_2m8g
# qwen2_5-vl-7b_cc3m_llm_stage1_model_lr4e-4_bsz60_epoch1_lorar32_alpha64_8m8g
# qwen2_5-vl-7b_cc3m_llm_stage1_model_lr6e-4_bsz60_epoch1_lorar32_alpha64_8m8g
# qwen2_5-vl-7b_cc3m_llm_stage1_model_lr6e-4_bsz60_epoch2_lorar32_alpha64_8m8g
# MODEL_NAMES=(
#     # "qwen2_5-vl-7b_cc3m_llm_stage1_model"
#     # "qwen2_5-vl-7b_cc3m_llm_stage1_model_lr2e-4_bsz60_epoch1_lorar32_alpha64_8m8g"
#     # "qwen2_5-vl-7b_cc3m_llm_stage1_model_lr4e-4_bsz60_epoch1_lorar32_alpha64_2m8g"
#     # "qwen2_5-vl-7b_cc3m_llm_stage1_model_lr4e-4_bsz60_epoch1_lorar32_alpha64_8m8g"
#     # "qwen2_5-vl-7b_cc3m_llm_stage1_model_lr6e-4_bsz60_epoch1_lorar32_alpha64_8m8g"
#     "qwen2_5-vl-7b_cc3m_llm_stage1_model_lr6e-4_bsz60_epoch2_lorar32_alpha64_8m8g"
# )

# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     echo "Evaluating model: ${MODEL_NAME}"
#     mkdir -p "./result/pretrain/${MODEL_NAME}/flickr/"
#     CUDA_VISIBLE_DEVICES='0,1' accelerate launch --multi_gpu --main_process_port 29500 eval/eval_zeroshot/eval_flickr.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --original_model_id ./checkpoints/qwen2_5-vl-7b_nli_pertrain_stage1_model_lr4e-4_bsz120_epoch2_lorar32_alpha64-merged \
#     --save_dir_name "./result/pretrain/${MODEL_NAME}/flickr/" \
#     --model_id ./checkpoints/${MODEL_NAME} \
#     >> ./result/pretrain/${MODEL_NAME}/flickr/pretrain_flickr.log
# done
# echo "All evaluations completed."




# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2_5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2_5-vl-7b_LamRA_Ret_Pretrain

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_coco.py \
#     --image_data_path ./data/coco/coco_images \
#     --text_data_path ./data/coco/coco_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2_5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2_5-vl-7b_LamRA_Ret_Pretrain
