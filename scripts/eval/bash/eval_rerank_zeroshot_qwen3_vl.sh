source /home/user_name/miniconda3/bin/activate && conda activate u-marvel_qwen3
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages:$PYTHONPATH
export PATH="/home/user_name/miniconda3/envs/u-marvel_qwen3/bin/:$PATH"

MODEL_NAME="qwen3-vl-4b_m-beir_stage1_model-Rank-Only-Pointwise"
SOURCE_MODEL_NAME="qwen3-vl-4b_m-beir_stage3_model"
MODEL_ID="./checkpoints/rerank_model/${MODEL_NAME}"
ORIGINAL_MODEL_ID="./checkpoints/hf_models/Qwen3-VL-4B-Instruct"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")


# TASK_NAME=urban1k_i2t
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/Urban1k/image \
#     --text_data_path ./data/Urban1k/caption \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/urban1k/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/urban1k/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=urban1k_t2i
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/Urban1k/image \
#     --text_data_path ./data/Urban1k/caption \
#     --model_id $MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/urban1k/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/urban1k/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sharegpt4v_i2t
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/sharegpt4v/val_data \
#     --text_data_path ./data/sharegpt4v/datas_for_validation.json \
#     --model_id $MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sharegpt4v_t2i
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/sharegpt4v/val_data \
#     --text_data_path ./data/sharegpt4v/datas_for_validation.json \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=flickr_i2t
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/flickr/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/flickr/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=flickr_t2i
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --image_data_path ./data/flickr/images \
#     --text_data_path ./data/flickr/flickr_text.json \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/flickr/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/flickr/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=genecis_change_object
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/genecis/annotations/change_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_object/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_object/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=genecis_focus_object
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/genecis/annotations/focus_object.json \
#     --image_path_prefix ./data/genecis/val2017 \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_object/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_object/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=genecis_change_attribute
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/genecis/annotations/change_attribute.json \
#     --image_path_prefix ./data/genecis/vg/change_attribute \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_attribute/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_attribute/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=genecis_focus_attribute
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/genecis/annotations/focus_attribute.json \
#     --image_path_prefix ./data/genecis/vg/focus_attribute \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_attribute/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_attribute/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

TASK_NAME=circo
PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
mkdir -p ${PATH_NAME_FIX}
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
    --annotation_path ./data/circo/annotations \
    --image_path_prefix ./data/circo/images/unlabeled2017 \
    --model_id $MODEL_ID \
    --original_model_id $ORIGINAL_MODEL_ID \
    --task_name ${TASK_NAME} \
    --batch_size 16 \
    --path_name_fix ${PATH_NAME_FIX} \
    --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/circo/${TASK_NAME}_test_query_names_val.json \
    --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/circo/${TASK_NAME}_test_cand_names_val.json \
    --rank_num 50 \
    --save_name ${TASK_NAME}_top50_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log


# TASK_NAME=visdial
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --data_path ./data/visdial/visdial_1.0_val.json \
#     --image_path_prefix ./data/visdial/VisualDialog_val2018 \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/visdial/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/visdial/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=mrf
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --data_path ./data/multiturnfashion/data/all.val.json \
#     --annotation_path ./data/multiturnfashion/image_splits/split.all.val.json \
#     --image_path_prefix ./data/M-BEIR/mbeir_images/fashioniq_images \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/multiturn_fashion/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/multiturn_fashion/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --batch_size 2 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=ccneg
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/ccneg/ccneg_preprocessed.pt \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type add_att \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type add_obj \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type replace_att \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --rank_num 2 \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type replace_obj \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type replace_rel \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type swap_att \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=sugar_crepe
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --annotation_path ./data/sugar-crepe/data \
#     --image_path_prefix ./data/sugar-crepe/images/val2017 \
#     --data_type swap_obj \
#     --model_id $MODEL_ID \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --save_name ${TASK_NAME}_top2_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log

# TASK_NAME=msvd_t2v
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --video_path_prefix ./data/MSVD/YouTubeClips \
#     --test_video_path ./data/MSVD/test_list.txt \
#     --captions_path ./data/MSVD/raw-captions.pkl \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/msvd/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/msvd/${TASK_NAME}_cand_names.json \
#     --rank_num 10 \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log


# TASK_NAME=msrvtt_t2v
# PATH_NAME_FIX=./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${TASK_NAME}/
# mkdir -p ${PATH_NAME_FIX}
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot_rerank_qwen3_vl.py \
#     --data_path ./data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv \
#     --video_data_path ./data/msrvtt/videos \
#     --original_model_id $ORIGINAL_MODEL_ID \
#     --model_id $MODEL_ID \
#     --task_name ${TASK_NAME} \
#     --path_name_fix ${PATH_NAME_FIX} \
#     --rank_num 10 \
#     --batch_size 2 \
#     --ret_query_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/msrvtt/${TASK_NAME}_query_names.json \
#     --ret_cand_data_path ./result/${SOURCE_MODEL_NAME}_eval_results_finetune/zeroshot/msrvtt/${TASK_NAME}_cand_names.json \
#     --save_name ${TASK_NAME}_top10_all > ./result/result_rank/${MODEL_NAME}/${SOURCE_MODEL_NAME}/zeroshot/${CURRENT_TIME}_rerank_${TASK_NAME}.log