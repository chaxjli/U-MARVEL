#!/bin/bash
model_names=(
    U-MARVEL-Qwen2VL-7B-Instruct
)

# 遍历模型名称列表,处理其他模型
for model_name in "${model_names[@]}"; do
    echo "Running evaluation for model: $model_name"
    python eval/eval_mbeir_global_debug.py --model_name "$model_name"
    
    # 检查 Python 脚本的返回状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while running evaluation for model: $model_name"
        exit 1
    fi
done

echo "All evaluations completed successfully."