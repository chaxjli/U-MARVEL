source /users/miniconda3/bin/activate && conda activate U-MARVEL_npu
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/users/miniconda3/envs/U-MARVEL_npu/bin/:$PATH"

MODEL_NAME="U-MARVEL-Qwen2VL-7B-Instruct"
ORIGINAL_MODEL_ID="./checkpoints/U-MARVEL-Qwen2VL-7B-Instruct"
MODEL_ID="./checkpoints/${MODEL_NAME}"

CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
mkdir -p "./result/${MODEL_NAME}_eval_results_finetune/zeroshot"

USE_INSTRUCTION_MASK="True"
MEAN_POOLING="True"
USE_BI_ATTEN="True"

# # 下面的数据集 zero-shot 评估结果均已经超过 paper 当中的结果 ----------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_urban1k.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --image_data_path ./data/Urban1k/image \
#     --text_data_path ./data/Urban1k/caption \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/urban1k" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_urban1k.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_ccneg.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --data_path ./data/ccneg/ccneg_preprocessed.pt \
#     --batch_size 8 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/ccneg" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_ccneg.log"

# # split val/test
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_circo.py \
    --use_instruction_mask $USE_INSTRUCTION_MASK \
    --mean_pooling $MEAN_POOLING \
    --use_bi_atten $USE_BI_ATTEN \
    --annotation_path_prefix ./data/circo/annotations \
    --image_path_prefix ./data/circo/images/unlabeled2017 \
    --split val \
    --batch_size 4 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/circo" \
    --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_circo.log"


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sharegpt4v.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --image_data_path ./data/sharegpt4v/val_data \
#     --text_data_path ./data/sharegpt4v/datas_for_validation.json \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v" \
#     --model_id $MODEL_ID \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sharegpt4v.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path ./data/genecis/annotations/change_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --data_type change_object \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_object" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_genecis_change_object.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path ./data/genecis/annotations/focus_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --data_type focus_object \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_object" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_genecis_focus_object.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path ./data/genecis/annotations/change_attribute.json \
#     --image_path_prefix ./data/genecis/vg/change_attribute \
#     --data_type change_attribute \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_attribute" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_genecis_change_attribute.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path ./data/genecis/annotations/focus_attribute.json \
#     --image_path_prefix ./data/genecis/vg/focus_attribute \
#     --data_type focus_attribute \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_attribute" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_genecis_focus_attribute.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/flickr" \
#     --mode finetuned \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_flickr.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type add_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_att" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_add_att.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type add_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_obj" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_add_obj.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_att" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_replace_att.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_obj" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_replace_obj.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type replace_rel \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_rel" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_replace_rel.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type swap_att \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_att" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_swap_att.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --annotation_path_prefix ./data/sugar-crepe/data \
#     --data_type swap_obj \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_obj" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_sugar_crepe_swap_obj.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_visdial.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --data_path ./data/visdial/visdial_1.0_val.json \
#     --image_path_prefix ./data/visdial/VisualDialog_val2018 \
#     --batch_size 4 \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/visdial" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_visdial.log"


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_multiturn_fashion.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --query_data_path ./data/multiturnfashion/data/all.val.json \
#     --cand_data_path ./data/multiturnfashion/image_splits/split.all.val.json \
#     --image_path_prefix ./data/M-BEIR/mbeir_images/fashioniq_images \
#     --batch_size 4 \
#     --category all \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/multiturn_fashion" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_multiturn_fashion.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msvd.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --video_path_prefix ./data/MSVD/YouTubeClips \
#     --test_video_path ./data/MSVD/test_list.txt \
#     --captions_path ./data/MSVD/raw-captions.pkl \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/msvd" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_msvd.log"

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msrvtt.py \
#     --use_instruction_mask $USE_INSTRUCTION_MASK \
#     --mean_pooling $MEAN_POOLING \
#     --use_bi_atten $USE_BI_ATTEN \
#     --data_path ./data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv \
#     --video_data_path ./data/msrvtt/videos \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --save_dir_name "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/msrvtt" \
#     --save_for_rerank > "./result/${MODEL_NAME}_eval_results_finetune/zeroshot/${CURRENT_TIME}_eval_msrvtt.log"



