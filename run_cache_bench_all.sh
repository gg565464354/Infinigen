#!/bin/bash
set -euo pipefail

ROOT="/root/InfiniGen"
AAA_DIR="$ROOT/speedup/flexgen/flexgen/aaa"
TARGET_DIR="$ROOT/speedup/flexgen/flexgen"
BENCH="${BENCH:-$ROOT/my_cache_bench.sh}"
LOG_DIR="$ROOT/log/switch"
LOG_FILE="$LOG_DIR/run_log.txt"
RESULTS="$LOG_DIR/throughput_table.csv"
RUNNER_LOG="$LOG_DIR/run_cache_bench_all.log"

if [ ! -d "$AAA_DIR" ]; then
  echo "Missing directory: $AAA_DIR" >&2
  exit 1
fi

if [ ! -f "$BENCH" ]; then
  echo "Missing file: $BENCH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if [ ! -f "$RESULTS" ]; then
  echo "version,bsz,total_throughput" > "$RESULTS"
fi

backup_dir="$LOG_DIR/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp "$TARGET_DIR/flex_opt.py" "$backup_dir/"
cp "$TARGET_DIR/pytorch_backend.py" "$backup_dir/"
cp "$TARGET_DIR/cache_selection_controller_v2.py" "$backup_dir/"

if [ "$#" -gt 0 ]; then
  versions=("$@")
else
  mapfile -t versions < <(find "$AAA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

zero_gpu_cache_versions=(
  "qwen_infinigen_cache_cpu"
  "qwen_quest_cache_cpu"
  "qwen_quest_straw"
)

needs_zero_gpu_cache() {
  local v="$1"
  for z in "${zero_gpu_cache_versions[@]}"; do
    if [ "$v" = "$z" ]; then
      return 0
    fi
  done
  return 1
}

set_gpu_cache_num() {
  local val="$1"
  if grep -q '^gpu_cache_num=' "$BENCH"; then
    sed -i "s/^gpu_cache_num=.*/gpu_cache_num=$val/" "$BENCH"
  else
    echo "gpu_cache_num=$val" >> "$BENCH"
  fi
}

for v in "${versions[@]}"; do
  src="$AAA_DIR/$v"
  if [ ! -d "$src" ]; then
    echo "Skip missing version dir: $src" >&2
    continue
  fi

  for f in flex_opt.py pytorch_backend.py cache_selection_controller_v2.py; do
    if [ ! -f "$src/$f" ]; then
      echo "Missing $src/$f" >&2
      continue 2
    fi
  done

  echo "===== Version: $v =====" | tee -a "$RUNNER_LOG"
  echo "$v" > "$LOG_DIR/current_version.txt"

  cp "$src/flex_opt.py" "$TARGET_DIR/flex_opt.py"
  cp "$src/pytorch_backend.py" "$TARGET_DIR/pytorch_backend.py"
  cp "$src/cache_selection_controller_v2.py" "$TARGET_DIR/cache_selection_controller_v2.py"

  if needs_zero_gpu_cache "$v"; then
    set_gpu_cache_num 0
  else
    set_gpu_cache_num 1
  fi

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1, skip bench for $v" | tee -a "$RUNNER_LOG"
    continue
  fi

  if ! (cd "$ROOT" && THROUGHPUT_CSV="$RESULTS" VERSION_TAG="$v" bash "$BENCH"); then
    echo "Bench failed for $v" | tee -a "$RUNNER_LOG" >&2
    continue
  fi
  echo "===== Version: $v DONE =====" | tee -a "$RUNNER_LOG"

done

echo "Done. Results: $RESULTS"
echo "Backup: $backup_dir"
echo "Runner log: $RUNNER_LOG"
