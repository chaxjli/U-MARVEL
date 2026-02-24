# 设置可见的 CUDA 设备，这里指定使用编号为 0 到 7 的共 8 块 GPU
source /home/user_name/miniconda3/bin/activate && conda activate u-marvel

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# 使用 accelerate 工具启动脚本，--multi_gpu 表示使用多 GPU 模式
# launch 命令用于启动脚本，并自动处理分布式训练的相关设置
# --main_process_port 29508 指定主进程的端口号为 29508，用于多进程之间的通信

accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \

    --image_data_path ./data/flickr/images \
    --text_data_path ./data/flickr/flickr_text.json \

    # --original_model_id 参数指定原始预训练模型的存储路径
    --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \

    # --model_id参数指定经过微调或训练后的模型的存储路径
    --model_id ./checkpoints/qwen2.5-vl-7b_LamRA_Ret_Pretrain

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_coco.py \
#     --image_data_path ./data/coco/coco_images \
#     --text_data_path ./data/coco/coco_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2.5-vl-7b_LamRA_Ret_Pretrain
