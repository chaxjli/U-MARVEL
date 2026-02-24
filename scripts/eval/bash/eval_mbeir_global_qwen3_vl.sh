#!/bin/bash
# conda activate env-3.8.8
# pip --default-time=1000 install -i https://pypi.tuna.tsinghua.edu.cn/simple faiss-gpu
# 镜像是：mirrors.tencent.com/todacc/venus-std-ext-cuda11.3-py3.8-tf2.7-pytorch1.11:0.2.1
# 评估 MBEIR 全局模型
# 用法: sh scripts/eval/bash/eval_mbeir_global_qwen3_vl.sh
# 定义模型名称列表
model_names=(

)

# 遍历模型名称列表,处理其他模型
for model_name in "${model_names[@]}"; do
    echo "Running evaluation for model: $model_name"
    python eval/eval_mbeir_global_debug_qwen3_vl.py --model_name "$model_name"
    
    # 检查 Python 脚本的返回状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while running evaluation for model: $model_name"
        exit 1
    fi
done

echo "All evaluations completed successfully."