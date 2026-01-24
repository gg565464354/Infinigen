#!/bin/bash
# 确保确定性执行

# 设置要测试的 gpu-batch-size 值列表
# batch_sizes=(24 25 26 27 28 29 30 31 32)
# batch_sizes=(12 13 14 15 16)
# batch_sizes=1
batch_sizes=(16 24 32)
prompt_len=8196
max_kv=2048
gen_len=512
# gen_len=10

# gpu_cache_num=30
gpu_cache_num=1


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/log/switch"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_log.txt"
THROUGHPUT_CSV="${THROUGHPUT_CSV:-$LOG_DIR/throughput_table.csv}"
if [ ! -f "$THROUGHPUT_CSV" ]; then
    echo "version,bsz,total_throughput" > "$THROUGHPUT_CSV"
fi
if [ -z "${VERSION_TAG:-}" ] && [ -f "$LOG_DIR/current_version.txt" ]; then
    VERSION_TAG="$(cat "$LOG_DIR/current_version.txt")"
fi
if [ -z "${VERSION_TAG:-}" ]; then
    AAA_DIR="$SCRIPT_DIR/speedup/flexgen/flexgen/aaa"
    TARGET_DIR="$SCRIPT_DIR/speedup/flexgen/flexgen"
    if [ -d "$AAA_DIR" ]; then
        for d in "$AAA_DIR"/*; do
            [ -d "$d" ] || continue
            if [ -f "$d/flex_opt.py" ] && [ -f "$d/pytorch_backend.py" ] && [ -f "$d/cache_selection_controller_v2.py" ]; then
                if cmp -s "$d/flex_opt.py" "$TARGET_DIR/flex_opt.py" \
                    && cmp -s "$d/pytorch_backend.py" "$TARGET_DIR/pytorch_backend.py" \
                    && cmp -s "$d/cache_selection_controller_v2.py" "$TARGET_DIR/cache_selection_controller_v2.py"; then
                    VERSION_TAG="$(basename "$d")"
                    break
                fi
            fi
        done
    fi
fi

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
        --gpu-cache-num "$gpu_cache_num" \
        --gpu-cache-pred 2 \
        --cpu-cache-pred 2 \
        2>&1 | tee -a "$LOG_FILE"

    if [ -n "${VERSION_TAG:-}" ]; then
        last_row=$(awk '
            /total throughput:/ {
                for (i = 1; i <= NF; i++) {
                    if ($i == "throughput:") {
                        t = $(i + 1)
                        sub(/[^0-9.].*/, "", t)
                    }
                }
            }
            /bsz:/ {
                for (i = 1; i <= NF; i++) {
                    if ($i == "bsz:") {
                        b = $(i + 1)
                        sub(/[^0-9.].*/, "", b)
                    }
                }
            }
            END {
                if (t != "" && b != "") {
                    print b "," t
                }
            }
        ' "$LOG_FILE")
        if [ -n "$last_row" ]; then
            echo "${VERSION_TAG},${last_row}" >> "$THROUGHPUT_CSV"
        else
            echo "WARN: failed to parse throughput for bsz=$batch_size" >&2
        fi
    elif [ -z "${THROUGHPUT_NOTICE_SHOWN:-}" ]; then
        echo "WARN: VERSION_TAG not set; skip CSV append" >&2
        THROUGHPUT_NOTICE_SHOWN=1
    fi

    echo -e "\n\n" | tee -a "$LOG_FILE"
done
