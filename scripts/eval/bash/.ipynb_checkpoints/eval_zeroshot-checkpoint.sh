source /home/user_name/miniconda3/bin/activate && conda activate u-marvel
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel/bin/:$PATH"

ORIGINAL_MODEL_ID="./checkpoints/hf_models/Qwen2-VL-7B-Instruct"
MODEL_ID="./checkpoints/qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt"
MODEL_NAME="qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_urban1k.py \
#     --image_data_path ./data/Urban1k/image \
#     --text_data_path ./data/Urban1k/caption \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/urban1k/" \
#     --save_for_rerank >> "./result/${MODEL_NAME}_eval_results/zeroshot/eval_urban1k.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_circo.py \
#     --annotation_path_prefix ./data/circo/annotations \
#     --image_path_prefix ./data/circo/images/unlabeled2017 \
#     --split test \
#     --batch_size 16 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_vist.py \
#     --data_path ./data/vist/sis/val.story-in-sequence.json \
#     --image_path_prefix ./data/vist/images/val \
#     --batch_size 2 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sharegpt4v.py \
#     --image_data_path ./data/sharegpt4v/val_data \
#     --text_data_path ./data/sharegpt4v/datas_for_validation.json \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/sharegpt4v/" \
#     --model_id $MODEL_ID \
#     --save_for_rerank >> "./result/${MODEL_NAME}_eval_results/zeroshot/eval_sharegpt4v.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/flickr/" \
#     --mode finetuned \
#     --save_for_rerank >> "./result/${MODEL_NAME}_eval_results/zeroshot/eval_flickr.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_ccneg.py \
#     --data_path ./data/ccneg/ccneg_preprocessed.pt \
#     --batch_size 8 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/ccneg/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_ccneg.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --annotation_path ./data/genecis/annotations/change_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --data_type change_object \
#     --batch_size 16 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/ccgenecis/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_genecis_change_object.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --annotation_path ./data/genecis/annotations/focus_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --data_type focus_object \
#     --batch_size 16 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/ccgenecis/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_genecis_focus_object.log"


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --annotation_path ./data/genecis/annotations/change_attribute.json \
#     --image_path_prefix ./data/genecis/vg/change_attribute \
#     --data_type change_attribute \
#     --batch_size 16 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/ccgenecis/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_genecis_change_attribute.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --annotation_path ./data/genecis/annotations/focus_attribute.json \
#     --image_path_prefix ./data/genecis/vg/focus_attribute \
#     --data_type focus_attribute \
#     --batch_size 16 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/ccgenecis/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_genecis_focus_attribute.log"

    
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_visdial.py \
#     --data_path ./data/visdial/visdial_1.0_val.json \
#     --image_path_prefix ./data/visdial/VisualDialog_val2018 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/visdial/" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_visdial.log"


CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_multiturn_fashion.py \
    --query_data_path ./data/multiturnfashion/data/all.val.json \
    --cand_data_path ./data/multiturnfashion/image_splits/split.all.val.json \
    --image_path_prefix ./data/M-BEIR/mbeir_images/fashioniq_images \
    --batch_size 4 \
    --category all \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_dir_name "./result/${MODEL_NAME}_eval_results/zeroshot/multiturnfashion/" \
    --save_for_rerank > "./result/${MODEL_NAME}_eval_results/zeroshot/eval_multiturn_fashion.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type add_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type add_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_rel \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type swap_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type swap_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_for_rerank

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msvd.py \
#     --video_path_prefix ./data/MSVD/YouTubeClips \
#     --test_video_path ./data/MSVD/test_list.txt \
#     --captions_path ./data/MSVD/raw-captions.pkl \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id code-kunkun/LamRA-Ret

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msrvtt.py \
#     --data_path ./data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv \
#     --video_data_path ./data/msrvtt/videos \
#     --original_model_id ./checkpoints/hf_models/Qwen2-VL-7B-Instruct \
#     --model_id code-kunkun/LamRA-Ret

