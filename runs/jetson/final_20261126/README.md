# Final M on Jetson Orin Nano 8GB — 2026-11-26

This directory records the board run of the final fine-tuned model. The user restored the
server backup to `/home/jetson/qwen_vlm/server_download`; every model/adapter file was
checked byte-for-byte against `runs/M_final/bundle_SHA256SUMS` before use.

## Restored assets

| Uploaded file | Bytes | SHA256 | Server manifest |
| --- | ---: | --- | --- |
| `final-q4_k_m.gguf` | 1,107,410,048 | `ebbb2b33…a52cbc` | match |
| `mmproj-final-f16.gguf` | 819,394,784 | `96671dad…0174d3` | match |
| `adapter/adapter_model.safetensors` | 69,788,264 | `db07fc40…e4368a` | match |
| `adapter/adapter_config.json` | 1,213 | `92c76cb2…09772e` | match |
| `adapter/README.md` | 5,280 | `7c6c0777…ada12a` | match |
| `session_001.zip` | 98,386,709 | `ac496627…ca83f2` | 50 PNG + 50 JSON, all 50 annotated paths covered |

`release/jetson_bundle_final/` was assembled from these files (Q4 only; Q8/F16 were not
uploaded) and has its own 86-file `SHA256SUMS`, copied here as `bundle_SHA256SUMS`.

## Commands

```bash
cd /home/jetson/qwen_vlm/release/jetson_bundle_base   # board-built b12425 + venv
GGUF=/home/jetson/qwen_vlm/release/jetson_bundle_final/models/final-q4_k_m.gguf \
MMPROJ=/home/jetson/qwen_vlm/release/jetson_bundle_final/models/mmproj-final-f16.gguf \
MODEL_TAG=final IMAGE=/home/jetson/qwen_vlm/data/raw/session_001/session_001_20261118_153745_750016_4e0334.png \
PIXELS=512 bash deploy/jetson/02_run_baseline.sh

MODEL_TAG=final MODELS_DIR=/home/jetson/qwen_vlm/release/jetson_bundle_final/models \
QUANTS=q4_k_m PIXELS=384,512,768 REPEAT=100 WARMUP=5 bash deploy/jetson/03_bench.sh

# real photos (100 checklist rows over 50 PNGs)
llama-server -m .../final-q4_k_m.gguf --mmproj .../mmproj-final-f16.gguf \
  --host 127.0.0.1 --port 8080 --jinja --temp 0 -c 4096 -ngl 99 -b 2048 -ub 2048 \
  --image-min-tokens 64 --image-max-tokens 256 --no-warmup --fit off
PYTHONPATH=src .venv/bin/python scripts/predict_llamacpp.py \
  --server http://127.0.0.1:8080 --jsonl data/annotations/real_v1.jsonl \
  --max-pixels 262144 --experiment-id jetson_final_q4_512 \
  --output runs/jetson/final_20261126/real_eval/predictions_real.jsonl
```

## Results

`summary.md` / `summary.json` contain the full tables. Key points:

- Final Q4_K_M, smoke-image benchmark: 3 pixel tiers × 105 requests, all parsed.
  Cold p50/p95 1.68/2.12, 1.86/2.29, 2.53/2.80 s at 384²/512²/768²; steady p50 1.21–1.29 s;
  37.9–38.9 token/s; peak RAM 7.22–7.31 GB; peak VDD_IN 19.3–19.9 W.
- Real-photo deployment check (100 checklist rows, 50 training photos):
  board decision 77/100, false-complete 2/31, unknown recall 16/18, state accuracy 446/500,
  JSON valid 100/100, end-to-end p50/p95 3.01/3.39 s.
  L40 reference on the same rows: 77/100, 2/31, 16/18, 439/500, 100/100.
  Per-sample agreement: 91/100 states, 96/100 decisions.
- Gradio saved-image mode on v001/v014/v040/v041: board decisions match the L40 results
  (unknown/incomplete/complete/unknown); board inference 3.16–3.57 s.

## Caveats

- The 50 real photos are part of final training. These metrics are a **deployment/export
  check**, not a generalization metric. The held-out three-fold result remains
  86/100, 2/31, 1/18.
- The board ran with the desktop session and this agent active; RAM/swap include that
  overhead. Swap used also carries state from the earlier base-weight benchmark.
- The smoke-image benchmark repeats three images, so its steady state hits the llama-server
  prompt cache. Real-photo requests use unique images and are the more realistic latency.
- Final Q8_0 and F16 GGUFs were not uploaded. F16 OOM evidence in `../board_20261125/`
  is from the base weights, not the final model.
- No `/dev/video*`: camera FPS, live demonstration and video remain pending.

## Tests

`PYTHONPATH=src python -m pytest -q tests` → **111 passed, 3 failed** (`tests.txt`).
The three failures are the same environment-specific ones as the 2026-11-25 run:
one test hardcodes the L40 server venv, and two training-label tests expect the
server's transformers 5.5.4 tokenizer return types. No source was changed.

Note: the board venv needed `imagehash` for `predict_llamacpp.py`; it is now listed in
`deploy/jetson/requirements.txt` and installed in the board bundle.

## Files

- `baseline/` — one cold request on v001 (final Q4 512)
- `bench/` — 3 smoke-image configs, server logs, tegrastats
- `real_eval/` — board predictions, tegrastats, eval metrics, L40 reference eval
- `demo/` — Gradio `/infer` outputs for the four real demo photos
- `app/` — 04 app/server logs
- `analyze_final.py` — regenerates `summary.json` / `summary.md`
- `tests.txt` — board test run (see the 2026-11-25 evidence for environment failures)
