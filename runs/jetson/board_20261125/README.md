# Jetson Orin Nano 8GB board run — 2026-11-25

This directory is the evidence for the board-side deployment work. It was produced on the
device itself (`/home/jetson/qwen_vlm`), not on the L40 server.

## What was run

| Step | Command | Result |
| --- | --- | --- |
| Probe | `bash deploy/jetson/00_probe.sh` | PASS. Orin Nano 8GB Super, L4T R36.4.3, CUDA 12.6.11, `MAXN_SUPER`, 7.4 GiB RAM + 11 GiB swap, no `/dev/video*` |
| Build | `bash deploy/jetson/01_build_llamacpp.sh` | PASS. llama.cpp b12425 (`18b74fa4`), CUDA sm_87, `llama-server` / `llama-mtmd-cli` / `llama-quantize` / `llama-bench` built with `-j 2`; board venv from `deploy/jetson/requirements.txt` |
| Baseline | `bash deploy/jetson/02_run_baseline.sh` | PASS. Q4_K_M, 512², cold end-to-end 2.679 s, prefill 0.982 s, 56 generated tokens, parsed |
| Bench | `bash deploy/jetson/03_bench.sh` | Q4_K_M and Q8_0 × 384²/512²/768²: 6 configs × (5 warm-up + 100 measured), all 600 measured requests parsed. F16: OOM-killed on first inference |
| App | `bash deploy/jetson/04_app.sh` | llama-server + Gradio on `:7860`; saved-image mode exercised through `/infer` on three smoke images |

`summary.md` is the human-readable table; `summary.json` has the full per-config data,
including tegrastats windows. `analyze_bench.py` regenerates both.

## Model provenance

The clone did not contain `release/`, `runs/M_final/adapter/` or `data/raw/`, so this run
used the **base** Qwen3-VL-2B-Instruct weights, not the final LoRA model:

- ModelScope `Qwen/Qwen3-VL-2B-Instruct-GGUF`, revision `1d49f1a70c0c1ea52f82fd46ea44b388183aa16c`
- `Qwen3VL-2B-Instruct-F16.gguf`, SHA256 `27b80f2e258f8372c9ee3a9b2293bd747cc08b18ed99709b57288cf457c42595`
- `mmproj-Qwen3VL-2B-Instruct-F16.gguf`, SHA256 `c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5`
- Q8_0 and Q4_K_M regenerated on the board with b12425 `llama-quantize`; their sizes match
  the official files exactly (1,834,427,424 and 1,107,409,952 bytes).
- Full hashes: `models_manifest.json`. The rebuilt bundle has a 100-file `SHA256SUMS`
  (copied here as `bundle_SHA256SUMS`); it covers the payload but not `.venv`,
  `llama.cpp/build` or `logs/`.

llama.cpp source: shallow clone of the `b12425` tag from the GitCode mirror
(`gitcode.com/gh_mirrors/ll/llama.cpp`), verified `git rev-parse HEAD` =
`18b74fa41dcac1f55ae1371fe58153f7225941f6`, repacked so the tarball top level is
`llama.cpp/`. The codeload GitHub tarball was too slow/unreliable from this network.

## Measured numbers (Q4_K_M / Q8_0)

| Quant | Pixels | Cold p50 (n=3) | Steady p50 / p95 (n=100) | Gen token/s | Peak RAM | Peak VDD_IN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 384² | 2.055 s | 1.616 / 1.670 s | 38.63 | 6.15 GB | 19.9 W |
| Q4_K_M | 512² | 2.297 s | 1.662 / 1.698 s | 37.88 | 6.25 GB | 20.2 W |
| Q4_K_M | 768² | 2.958 s | 1.709 / 1.741 s | 37.48 | 6.49 GB | 20.4 W |
| Q8_0 | 384² | 2.492 s | 1.983 / 2.083 s | 30.96 | 6.86 GB | 20.4 W |
| Q8_0 | 512² | 2.706 s | 2.012 / 2.115 s | 30.64 | 6.96 GB | 21.0 W |
| Q8_0 | 768² | 3.221 s | 2.046 / 2.073 s | 30.41 | 7.20 GB | 20.5 W |

F16: 3.21 GiB language weights + 0.76 GiB mmproj; `llama-server` loaded and answered health
checks, then the kernel OOM-killed it on the first inference (`f16_oom_kernel.log`). The
remaining F16 attempts were stopped to protect the session; no F16 numbers exist.

## How to read the timings

- **Cold** is the first presentation of each of the three smoke images (the warm-up requests
  with `attempt0_prompt_tokens > 100`), so n=3. This is the closest available proxy for a
  new photograph on the board.
- **Steady** is the 100 measured requests. They cycle three images, so after the first pass
  llama-server reuses the prompt cache; prefill drops to about 0.03 s and `total_s` is mostly
  generation. It is a sustained-throughput number, not a fresh-image latency.
- Per-config RAM/power windows are taken from the JSONL birth-to-last-write window and
  exclude model load. They include the desktop and agent processes, because the 8 GB is
  shared. The global tegrastats file covers all nine attempted configs.
- The inputs are `data/smoke/b0/*.png` (synthetic). These numbers are for deployment and
  performance only; no accuracy is claimed from them.

## Demo

`demo/image_mode_check.json` records three Gradio `/infer` calls (Q4_K_M, 512²):
`bench_hammer_screwdriver.png` → incomplete (missing cable), `bench_multimeter.png` → complete,
`bench_empty.png` → unknown. Wall time 2.15–2.40 s, all responses parsed. There is no camera
device, so the OpenCV/browser-camera paths and camera FPS were not tested.

## Tests on the board

`PYTHONPATH=src python -m pytest -q tests` → **111 passed, 3 failed** (full output in
`tests.txt`). The failures are environment-specific, not source regressions:

- `test_eval_report.py::test_eval_predictions_cli_writes_the_hand_counted_rates` hardcodes
  `/home/jetson/qwen_vlm/envs/qwen_vlm/bin/python` and `source env/activate.sh` from the L40
  server; that environment does not exist here (exit 127).
- The two `test_training_labels.py` cases expect the tokenizer tensors from the server's
  transformers 5.5.4; the board test venv used transformers 4.57, which returns lists.

No source files were changed for the board run.

## Missing for the full course deliverable

1. `runs/M_final/adapter/` or `release/jetson_bundle_final/` — restore from the cloud backup,
   then rerun `02`–`04` with `MODEL_TAG=final`.
2. `data/raw/session_001/` — restore the 50 real photos, then run `predict_llamacpp.py` on the
   board and evaluate the returned JSONL. For the final model these photos are training data;
   the result is a deployment check, not a generalization metric.
3. A USB camera — `/dev/video*` is absent; camera FPS and the recorded demo are still pending.
