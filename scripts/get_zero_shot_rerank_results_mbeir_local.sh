# get pointwise reranking results
# MODEL_NAME 替换成第一阶段产生数据的模型
MODEL_NAME="qwen2-vl-7b_BiLamRA_Ret_cc3m_llm_focal_infonce_loss_8m8g_4xlr"
ZERO_SHOT_RERANK_MODEL_NAME="Qwen2-VL-7B-Instruct"

python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name visualnews_task0 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
# python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name mscoco_task0 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name fashion200k_task0 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name webqa_task1 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name edis_task2 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name webqa_task2 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name visualnews_task3 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name mscoco_task3 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name fashion200k_task3 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name nights_task4 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name infoseek_task6 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name fashioniq_task7 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name cirr_task7 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name oven_task8 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name infoseek_task8 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME
python eval/rerank/mbeir_zero_shot_rerank_pointwise_local.py --task_name oven_task6 --model_name $MODEL_NAME --zero_shot_rerank_model_name $ZERO_SHOT_RERANK_MODEL_NAME




# # get listwise reranking results
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name visualnews_task0
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name mscoco_task0
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name fashion200k_task0
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name webqa_task1
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name edis_task2
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name webqa_task2
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name visualnews_task3
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name mscoco_task3
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name fashion200k_task3
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name nights_task4
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name oven_task6
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name infoseek_task6
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name fashioniq_task7
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name cirr_task7
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name oven_task8
# python eval/rerank/mbeir_rerank_listwise_comment.py --task_name infoseek_task8