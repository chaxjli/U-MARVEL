

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29500 eval/eval_zeroshot/eval_flickr.py \
    --image_data_path ./data/flickr/images \
    --text_data_path ./data/flickr/flickr_text.json \
    --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
    --model_id ./author_model/LamRA-Ret-Pretrained


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2.5-vl-7b_LamRA_Ret_Pretrain

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_coco.py \
#     --image_data_path ./data/coco/coco_images \
#     --text_data_path ./data/coco/coco_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2.5-vl-7b_LamRA_Ret_Pretrain
