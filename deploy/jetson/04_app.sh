#!/usr/bin/env bash
# 启动 llama-server 和演示界面。模型与依赖都在本机，启动之后可以断网。
# 环境变量：GGUF、MMPROJ、PIXELS（默认 512）、CAMERA（默认 0）、PORT、APP_PORT
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ./lib.sh
jetson_begin_log "04_app"

fail() {
  echo "RESULT: FAIL $*"
  exit 1
}

ROOT="$(jetson_root)"
MODEL_TAG="${MODEL_TAG:-base}"
GGUF="${GGUF:-$ROOT/models/$MODEL_TAG-q4_k_m.gguf}"
MMPROJ="${MMPROJ:-$ROOT/models/mmproj-$MODEL_TAG-f16.gguf}"
PIXELS="${PIXELS:-512}"
CAMERA="${CAMERA:-0}"
PORT="${PORT:-8080}"
APP_PORT="${APP_PORT:-7860}"
[[ -f "$GGUF" ]] || fail "找不到 GGUF：$GGUF"
[[ -f "$MMPROJ" ]] || fail "找不到 mmproj：$MMPROJ"
[[ -x "$ROOT/.venv/bin/python" ]] || fail "没有 $ROOT/.venv。先跑 01_build_llamacpp.sh"

SERVER_PID=""
APP_PID=""
cleanup() {
  jetson_stop_pid "$APP_PID"
  jetson_stop_pid "$SERVER_PID"
}
trap cleanup EXIT INT TERM

server_log="$ROOT/logs/llama_server_app_$(date +%Y%m%d_%H%M%S).txt"
SERVER_PID="$(jetson_start_server "$GGUF" "$MMPROJ" "$PIXELS" "$PORT" "$server_log")"
jetson_wait_health "$PORT" "$SERVER_PID" || fail "llama-server 没有起来，看 $server_log"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRADIO_ANALYTICS_ENABLED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
app_log="$ROOT/logs/app_$(date +%Y%m%d_%H%M%S).txt"
"$ROOT/.venv/bin/python" -m qwen_vlm.app.app \
  --server "http://127.0.0.1:$PORT" \
  --pixels "$PIXELS" \
  --camera "$CAMERA" \
  --host 0.0.0.0 \
  --port "$APP_PORT" \
  >"$app_log" 2>&1 &
APP_PID=$!

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    fail "演示界面进程退出，看 $app_log"
  fi
  if python3 - "$APP_PORT" 2>/dev/null <<'PY'
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/", timeout=2).read(32)
PY
  then
    ready=1
    break
  fi
  sleep 1
done
if ((ready != 1)); then
  fail "演示界面没有在 60 秒内打开，看 $app_log"
fi

echo "演示界面 http://0.0.0.0:$APP_PORT"
echo "llama-server 日志 $server_log"
echo "界面日志 $app_log"
echo "按 Ctrl+C 结束。断网不影响已经加载的模型。"
echo "RESULT: PASS"
wait "$APP_PID"
