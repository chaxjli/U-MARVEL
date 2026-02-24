# conda activate env-3.8.8
# pip --default-time=1000 install -i https://pypi.tuna.tsinghua.edu.cn/simple faiss-gpu
#!/bin/bash

# 定义模型名称列表
model_names=(
    "qwen2-vl-7b_BiLamRA-Ret"
    "qwen2-vl-7b_BiLamRA-RetWithMeanPooling"
    "qwen2-vl-7b_LamRA-RetWithMeanPooling"
    # "qwen2-vl-7b_LamRA-Ret_CVProj"
    # "qwen2-vl-7b_LamRA_Ret_LatentBlock"
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