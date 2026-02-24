import subprocess
import time
import os
import sys
import pathlib
import datetime
import shlex
# ==== NPU 环境配置 ====
# 激活虚拟环境, 这样激活虚拟环境只会使得子模块在该虚拟环境当中进行，主模块仍然在原来的环境当中运行
# os.system("source /home/user_name/miniconda3/bin/activate && conda activate u-marvel")
# os.environ["ASCEND_LAUNCH_BLOCKING"]="1"  # debug 时才开启，明白了

ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
os.environ["ASCEND_HOME_PATH"] = f'''{ASCEND_TOOLKIT_HOME}'''
os.environ["ASCEND_OPP_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/opp'''
os.environ["ASCEND_AICPU_PATH"] = f'''{ASCEND_TOOLKIT_HOME}'''
os.environ["TOOLCHAIN_HOME"] = f'''{ASCEND_TOOLKIT_HOME}/toolkit'''

ori_env_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/lib64/:{ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:{ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:{ori_env_path}'''
ori_env_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f'''{ASCEND_TOOLKIT_HOME}/tools/aml/lib64:{ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:{ori_env_path}'''

ori_env_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f'''{ASCEND_TOOLKIT_HOME}/python/site-packages:{ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:{ori_env_path}'''


ori_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = f'''{ASCEND_TOOLKIT_HOME}/bin:{ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:{ori_env_path}'''

ori_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = f'''/home/user_name/miniconda3/envs/u-marvel/bin/:{ori_env_path}'''

# 定义执行命令的函数
def run_command(command, log_file):
    with open(log_file, 'w') as f:
        try:
            result = subprocess.run(command, stdout=f, stderr=None, check=True,shell=True)
            print(f"Command {command} executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Command {command} failed with error: {e}")

MODEL_NAMES = [
            "qwen2-vl-7b_umarvel_distillation",
            ]

ORIGINAL_MODEL_IDS = [
            "qwen2-vl-7b_umarvel_hard_negative_mining-merged",
    ]
MODEL_IDS = [f"./checkpoints/{model_name}" for model_name in MODEL_NAMES]

for MODEL_NAME, ORIGINAL_MODEL_ID, MODEL_ID in zip(MODEL_NAMES, ORIGINAL_MODEL_IDS, MODEL_IDS):
    # 获取当前时间
    CURRENT_TIME = time.strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot", exist_ok=True)
    if MODEL_NAME == "qwen2-vl-7b_LamRA-Ret_author" or MODEL_NAME == "qwen2-vl-7b_LamRA-Ret":
        USE_INSTRUCTION_MASK = "False"
        MEAN_POOLING = "False"
        USE_BI_ATTEN = "False"
    else:    
        USE_INSTRUCTION_MASK = "True"
        MEAN_POOLING = "True"
        USE_BI_ATTEN = "True"
    CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
    # 执行 eval_urban1k.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_urban1k.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_urban1k.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --image_data_path ./data/Urban1k/image \
        --text_data_path ./data/Urban1k/caption \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/urban1k" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/urban1k/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sharegpt4v.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sharegpt4v.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sharegpt4v.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --image_data_path ./data/sharegpt4v/val_data \
        --text_data_path ./data/sharegpt4v/datas_for_validation.json \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v" \
        --model_id {MODEL_ID} \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sharegpt4v/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_flickr.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_flickr.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --image_data_path ./data/flickr/images \
        --text_data_path ./data/flickr/flickr_text.json \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/flickr" \
        --mode finetuned \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/flickr/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_ccneg.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_ccneg.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_ccneg.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --data_path ./data/ccneg/ccneg_preprocessed.pt \
        --batch_size 8 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccneg" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccneg/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_genecis.py (change_object) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_genecis_change_object.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path ./data/genecis/annotations/change_object.json \
        --image_path_prefix ./data/genecis/val2017 \
        --data_type change_object \
        --batch_size 2 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_object" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_object/{MODEL_NAME}.txt"
    if os.path.exists(checkfile) or MODEL_NAME == "qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt":
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_genecis.py (focus_object) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_genecis_focus_object.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path ./data/genecis/annotations/focus_object.json \
        --image_path_prefix ./data/genecis/val2017 \
        --data_type focus_object \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_object" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_object/{MODEL_NAME}.txt"
    if os.path.exists(checkfile) or MODEL_NAME == "qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt":
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_genecis.py (change_attribute) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_genecis_change_attribute.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path ./data/genecis/annotations/change_attribute.json \
        --image_path_prefix ./data/genecis/vg/change_attribute \
        --data_type change_attribute \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_attribute" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_change_attribute/{MODEL_NAME}.txt"
    if os.path.exists(checkfile) or MODEL_NAME == "qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt":
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务 .")
        run_command(command, log_file)

    # 执行 eval_genecis.py (focus_attribute) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_genecis_focus_attribute.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path ./data/genecis/annotations/focus_attribute.json \
        --image_path_prefix ./data/genecis/vg/focus_attribute \
        --data_type focus_attribute \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_attribute" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/ccgenecis_focus_attribute/{MODEL_NAME}.txt"
    if os.path.exists(checkfile) or MODEL_NAME == "qwen2-vl-7b_BiLamRA-RetWithMeanPooling_Withoutprompt_and_PretrainedWithoutprompt":
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_visdial.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_visdial.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_visdial.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --data_path ./data/visdial/visdial_1.0_val.json \
        --image_path_prefix ./data/visdial/VisualDialog_val2018 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/visdial" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/visdial/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_multiturn_fashion.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_multiturn_fashion.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_multiturn_fashion.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --query_data_path ./data/multiturnfashion/data/all.val.json \
        --cand_data_path ./data/multiturnfashion/image_splits/split.all.val.json \
        --image_path_prefix ./data/M-BEIR/mbeir_images/fashioniq_images \
        --batch_size 4 \
        --category all \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/multiturn_fashion" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/multiturn_fashion/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (add_att) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_add_att.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type add_att \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_att" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_att/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (add_obj) ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_add_obj.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type add_obj \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_obj" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_add_obj/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (replace_att) -----------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_replace_att.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type replace_att \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_att" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_att/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (replace_obj) -----------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_replace_obj.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type replace_obj \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_obj" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_obj/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (replace_rel) -----------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_replace_rel.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type replace_rel \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_rel" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_replace_rel/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (swap_att) -----------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_swap_att.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type swap_att \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_att" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_att/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_sugar_crepe.py (swap_obj) -----------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_sugar_crepe_swap_obj.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/sugar-crepe/data \
        --data_type swap_obj \
        --image_path_prefix ./data/sugar-crepe/images/val2017 \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_obj" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/sugar_crepe_swap_obj/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_msvd.py
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_msvd.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msvd.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --video_path_prefix ./data/MSVD/YouTubeClips \
        --test_video_path ./data/MSVD/test_list.txt \
        --captions_path ./data/MSVD/raw-captions.pkl \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/msvd" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/msvd/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_msrvtt.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_msrvtt.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_msrvtt.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --data_path ./data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv \
        --video_data_path ./data/msrvtt/videos \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/msrvtt" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/msrvtt/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)

    # 执行 eval_circo.py ---------------------------------------------------------------------------------------------------------------------------------------
    log_file = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/{CURRENT_TIME}_eval_circo.log"
    command = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_circo.py \
        --use_instruction_mask {USE_INSTRUCTION_MASK} \
        --mean_pooling {MEAN_POOLING} \
        --use_bi_atten {USE_BI_ATTEN} \
        --annotation_path_prefix ./data/circo/annotations \
        --image_path_prefix ./data/circo/images/unlabeled2017 \
        --split val \
        --batch_size 4 \
        --original_model_id {ORIGINAL_MODEL_ID} \
        --model_id {MODEL_ID} \
        --save_dir_name "./result/{MODEL_NAME}_eval_results_finetune/zeroshot/circo" \
        --save_for_rerank'
    checkfile = f"./result/{MODEL_NAME}_eval_results_finetune/zeroshot/circo/{MODEL_NAME}.txt"
    if os.path.exists(checkfile):
        print(f"File {checkfile} already exists. 跳过这个任务.")
    else:
        print(f"File {checkfile} does not exist. 执行这个任务.")
        run_command(command, log_file)