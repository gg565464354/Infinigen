#!/bin/bash
# 确保确定性执行

# 设置要测试的 gpu-batch-size 值列表
# batch_sizes=(24 25 26 27 28 29 30 31 32)
# batch_sizes=(12 13 14 15 16)
batch_sizes=(1 2 3 4)
# batch_sizes=(8)
prompt_len=8196
max_kv=2048
gen_len=512

# gpu_cache_num=30
gpu_cache_num=0


# 日志文件名
LOG_FILE="./log/switch/run_log.txt"

# 清空旧日志（如果存在）
> "$LOG_FILE"

# # 循环执行
for batch_size in "${batch_sizes[@]}"; do
    echo "===== Running with --prompt-len $prompt_len --max-num-kv $max_kv --gpu-batch-size $batch_size =====" | tee -a "$LOG_FILE"
    
    TRANSFORMERS_OFFLINE=1 python -m flexgen.flex_opt \
        --model /root/autodl-tmp/Qwen3-8B \
        --path /root/autodl-tmp/Qwen3-8B \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size "$batch_size" \
        --num-gpu-batches 1 \
        --prompt-len "$prompt_len" \
        --gen-len "$gen_len" \
        --warmup-input-path /root/InfiniGen/speedup/flexgen/pg19_firstbook.txt \
        --test-input-path /root/InfiniGen/speedup/flexgen/pg19_firstbook.txt \
        --alpha 4 \
        --partial-weight-ratio 0.2 \
        --max-num-kv "$max_kv" \
        --gpu-cache-num 0 \
        --gpu-cache-pred 2 \
        --cpu-cache-pred 2 \
        2>&1 | tee -a "$LOG_FILE"
    
    echo -e "\n\n" | tee -a "$LOG_FILE"
done
