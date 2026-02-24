# # get zeroshot rerank results
RERANKER_NAME="qwen3-vl-4b_m-beir_stage1_model-Rank-Only-Pointwise"
SOURCE_MODEL="qwen3-vl-4b_m-beir_stage3_model"
PATH_TO_SAVE="./result/result_rank/${RERANKER_NAME}/${SOURCE_MODEL}/zeroshot/merge_retrieval_rerank_results/log/"
# mkdir -p $PATH_TO_SAVE

python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name urban1k_t2i >> ${PATH_TO_SAVE}/urban1k_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name urban1k_i2t >> ${PATH_TO_SAVE}/urban1k_i2t.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name sharegpt4v_t2i >> ${PATH_TO_SAVE}/sharegpt4v_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name sharegpt4v_i2t >> ${PATH_TO_SAVE}/sharegpt4v_i2t.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name flickr_t2i >> ${PATH_TO_SAVE}/flickr_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name flickr_i2t >> ${PATH_TO_SAVE}/flickr_i2t.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_change_attribute >> ${PATH_TO_SAVE}/genecis_change_attribute.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_focus_attribute >> ${PATH_TO_SAVE}/genecis_focus_attribute.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_change_object >> ${PATH_TO_SAVE}/genecis_change_object.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_focus_object >> ${PATH_TO_SAVE}/genecis_focus_object.log
python eval/rerank/zeroshot_rerank/eval_visdial.py >> ${PATH_TO_SAVE}/visdial.log
python eval/rerank/zeroshot_rerank/eval_mrf.py --task_name mrf >> ${PATH_TO_SAVE}/mrf.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name ccneg >> ${PATH_TO_SAVE}/ccneg.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type add_att >> ${PATH_TO_SAVE}/sugar_crepe_add_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type add_obj >> ${PATH_TO_SAVE}/sugar_crepe_add_obj.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_att >> ${PATH_TO_SAVE}/sugar_crepe_replace_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_obj >> ${PATH_TO_SAVE}/sugar_crepe_replace_obj.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_rel >> ${PATH_TO_SAVE}/sugar_crepe_replace_rel.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type swap_att >> ${PATH_TO_SAVE}/sugar_crepe_swap_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type swap_obj >> ${PATH_TO_SAVE}/sugar_crepe_swap_obj.log
python eval/rerank/zeroshot_rerank/eval_circo.py  >> ${PATH_TO_SAVE}/circo.log
python eval/rerank/zeroshot_rerank/eval_msvd.py >> ${PATH_TO_SAVE}/msvd.log
python eval/rerank/zeroshot_rerank/eval_msrvtt.py >> ${PATH_TO_SAVE}/msrvtt.log
# get circo test submission file for evaluation