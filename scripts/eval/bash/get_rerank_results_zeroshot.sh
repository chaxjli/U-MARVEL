# # get zeroshot rerank results 
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name urban1k_t2i >> ./zeroshot_rerank_files/urban1k_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name urban1k_i2t >> ./zeroshot_rerank_files/urban1k_i2t.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name sharegpt4v_t2i >> ./zeroshot_rerank_files/sharegpt4v_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name sharegpt4v_i2t >> ./zeroshot_rerank_files/sharegpt4v_i2t.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name flickr_t2i >> ./zeroshot_rerank_files/flickr_t2i.log
python eval/rerank/zeroshot_rerank/eval_t2i_and_i2t.py --task_name flickr_i2t >> ./zeroshot_rerank_files/flickr_i2t.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_change_attribute >> ./zeroshot_rerank_files/genecis_change_attribute.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_focus_attribute >> ./zeroshot_rerank_files/genecis_focus_attribute.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_change_object >> ./zeroshot_rerank_files/genecis_change_object.log
python eval/rerank/zeroshot_rerank/eval_genecis.py --task_name genecis_focus_object >> ./zeroshot_rerank_files/genecis_focus_object.log
python eval/rerank/zeroshot_rerank/eval_visdial.py >> ./zeroshot_rerank_files/visdial.log
python eval/rerank/zeroshot_rerank/eval_mrf.py --task_name mrf >> ./zeroshot_rerank_files/mrf.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name ccneg >> ./zeroshot_rerank_files/ccneg.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type add_att >> ./zeroshot_rerank_files/sugar_crepe_add_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type add_obj >> ./zeroshot_rerank_files/sugar_crepe_add_obj.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_att >> ./zeroshot_rerank_files/sugar_crepe_replace_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_obj >> ./zeroshot_rerank_files/sugar_crepe_replace_obj.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type replace_rel >> ./zeroshot_rerank_files/sugar_crepe_replace_rel.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type swap_att >> ./zeroshot_rerank_files/sugar_crepe_swap_att.log
python eval/rerank/zeroshot_rerank/eval_itm.py --task_name sugar_crepe --data_type swap_obj >> ./zeroshot_rerank_files/sugar_crepe_swap_obj.log
python eval/rerank/zeroshot_rerank/eval_circo.py  >> ./zeroshot_rerank_files/circo.log
python eval/rerank/zeroshot_rerank/eval_msvd.py >> ./zeroshot_rerank_files/msvd.log
python eval/rerank/zeroshot_rerank/eval_msrvtt.py >> ./zeroshot_rerank_files/msrvtt.log
# get circo test submission file for evaluation