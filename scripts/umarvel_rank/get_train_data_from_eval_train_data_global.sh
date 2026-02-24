#!/bin/bash
# conda activate env-3.8.8
# pip --default-time=1000 install -i https://pypi.tuna.tsinghua.edu.cn/simple faiss-gpu
# 评估 MBEIR 全局模型
# 用法: sh scripts/lamra_rank/get_train_data_from_eval_train_data_global.sh
# 定义模型名称列表
model_names=(
    qwen2-vl-7b_umarvel_progressive_transition_m-beir
)

# 遍历模型名称列表,处理其他模型
for model_name in "${model_names[@]}"; do
    echo "Running evaluation for model: $model_name"
    python train/mbeir_get_rerank_train_data_from_eval_train_data_global.py --model_name "$model_name"
    
    # 检查 Python 脚本的返回状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while running evaluation for model: $model_name"
        exit 1
    fi
done

echo "All evaluations completed successfully."