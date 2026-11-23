# Jetson Orin Nano 部署说明

服务器不能 SSH 到板子。把本目录所在的 bundle 拷到板子上，按顺序自己跑脚本，再把 `logs/` 拷回服务器的 `runs/jetson/`。

脚本不用 `sudo`，不改系统包，也不改 `nvpmodel` / `jetson_clocks`。假设系统是 JetPack 6.x（L4T R36）、CUDA 12.6、Python 3.10。llama.cpp 与服务器是同一个 commit，见 `LLAMACPP_VERSION.md`。

## 1. 把 bundle 拷到板子

bundle 在服务器上的位置是 `release/jetson_bundle_base/`。两种办法：

- 网络：`scp -r jetson_bundle_base user@jetson:~/`
- 没有网：用 U 盘拷整个目录。拷完后在板子上核对 `SHA256SUMS`：

```bash
cd jetson_bundle_base
sha256sum -c SHA256SUMS
```

目录里有 llama.cpp 源码包、Q4_K_M 语言 GGUF、F16 的 mmproj、演示代码和三张冒烟图片。语言部分的 Q4 不是整个模型都是 INT4：视觉 mmproj 仍是 F16。

## 2. 按顺序跑

在 bundle 根目录下执行。每条脚本都会把完整输出写到 `logs/<脚本名>_<时间戳>.txt`，最后一行是 `RESULT: PASS` 或 `RESULT: FAIL` 加原因。

```bash
bash deploy/jetson/00_probe.sh
bash deploy/jetson/01_build_llamacpp.sh
bash deploy/jetson/02_run_baseline.sh
bash deploy/jetson/03_bench.sh
bash deploy/jetson/04_app.sh
```

使用最终模型的 bundle 时，为 `02`–`04` 设定 `MODEL_TAG=final`，脚本会读取 `final-*.gguf` 和 `mmproj-final-f16.gguf`。

`00_probe.sh` 只读：型号、L4T、`nvpmodel -q`、CUDA、内存和 swap、磁盘、Python、cmake/gcc、`/dev/video*`、`v4l2-ctl`、用户组、能不能访问 PyPI。

`01_build_llamacpp.sh` 解压源码并编译：

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF
```

然后用 `python3 -m venv .venv` 安装 gradio、opencv、pillow、pyyaml、jsonschema、requests。这一步需要网络。编译默认 `-j 2`，8GB 内存上不要把 `JOBS` 开得很大。需要 `nvcc` 在 `PATH` 里，以及系统自带的 `python3-venv`（脚本不会自己 apt 安装）。

`02_run_baseline.sh` 用 Q4_K_M、mmproj、默认 512² 对一张测试图推理。换档：

```bash
PIXELS=384 GGUF=models/base-q4_k_m.gguf bash deploy/jetson/02_run_baseline.sh
```

`03_bench.sh` 对量化档 × 像素档做计时。默认预热 5 次、正式 100 次，图片在三张冒烟图上循环，prompt 和 `max_new_tokens` 固定。后台跑 `tegrastats --interval 500`。每次请求写成 JSONL，字段与 `contracts/runner_api.md` 的预测格式一致。某一档 OOM 会记到 `logs/bench_*/errors.jsonl`，然后继续下一档。只跑一档可以：

```bash
QUANTS=q4_k_m PIXELS=512 REPEAT=100 bash deploy/jetson/03_bench.sh
```

默认 `QUANTS=q4_k_m,q8_0,f16`。本 bundle 如果只带了 Q4，另外两档会记成缺文件，整次脚本以 FAIL 结束，但 Q4 的 JSONL 仍然在。正式对比三档时，把对应 GGUF 放进 `models/`，名字是 `base-q8_0.gguf`、`base-f16.gguf`，mmproj 仍用 `mmproj-base-f16.gguf`。

测真实图片的点检准确率时，按原路径将 `data/annotations/real_v1.jsonl` 和 `data/raw/session_001/` 拷到 bundle，在板上的 llama-server 处于运行状态时按同一像素预算发请求：

```bash
PYTHONPATH=src .venv/bin/python scripts/predict_llamacpp.py \
  --server http://127.0.0.1:8080 --jsonl data/annotations/real_v1.jsonl \
  --max-pixels 262144 --experiment-id jetson_final_q4_512 \
  --output logs/jetson_final_q4_512_predictions.jsonl
```

同时采集 `tegrastats`，记录与这批请求对应的日志范围。回传预测 JSONL 和日志后，使用服务器的金标准计算板端指标。

`04_app.sh` 同时拉起 llama-server 和网页。浏览器打开 `http://<板子IP>:7860`。模型已经在本机，起来之后可以断网。摄像头设备号用 `CAMERA=0`（即 `/dev/video0`）。没有摄像头时，网页里选「读取保存图片」。

如果要在板子的本地显示器上演示“实时画面 + 空格触发推理”，用 `scripts/live_check.py`。它连接同一个 llama-server，用法见 `docs/live_check.md`。预打包的 bundle 生成早于该脚本，把 `scripts/live_check.py` 拷进 bundle 的 `scripts/` 即可运行。

## 3. 传回服务器的文件

请把这些拷回服务器的 `runs/jetson/`，不要改名：

- `logs/00_probe_*.txt`
- `logs/01_build_llamacpp_*.txt`
- `logs/02_run_baseline_*.txt` 和 `logs/baseline_*.jsonl`
- `logs/bench_*/` 整目录（含 tegrastats、nvpmodel、jsonl）
- `logs/04_app_*.txt` 和 `logs/llama_server_app_*.txt`（演示能打开即可，不必长时间挂着）

这些日志里的耗时和显存才是板端数字。服务器上的冒烟结果不能拿来代替。

## 4. 常见问题

- 显存不够：先把 `PIXELS` 降到 384，或确认用的是 Q4_K_M。F16 语言权重大约 3–4GB，加上 mmproj 和 KV，8GB 统一内存可能放不下。脚本失败时看 `server_*.log` 里有没有 `out of memory` 或 `cudaMalloc`。
- swap 不够：`00_probe.sh` 会打印 swap。编译或加载 F16 时如果被系统杀掉，需要一块 swap。脚本不会自己创建 swap。
- 摄像头权限：`groups` 里没有 `video` 时，`/dev/video*` 会打不开。脚本不用 sudo。可以改用网页里的「读取保存图片」，或者请管理员把用户加入 `video` 组。
- `python3 -m venv` 失败：系统没有 `ensurepip` / `python3-venv`。这需要管理员装一次，脚本不会装。
- 改了网页上的像素档但没重启 llama-server：视觉 token 上限是服务启动参数。`04_app.sh` 的 `PIXELS` 必须和当时启动的服务一致。
