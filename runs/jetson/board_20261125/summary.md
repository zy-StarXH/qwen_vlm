# Jetson Orin Nano 8GB 板端实测（2026-11-25）

- 设备：Jetson Orin Nano 8GB Engineering Reference Developer Kit Super，L4T R36.4.3，CUDA 12.6.11，MAXN_SUPER
- llama.cpp：b12425 / 18b74fa4，CUDA 编译（sm_87）
- 模型：Qwen3-VL-2B-Instruct 原始权重，官方 F16 GGUF（ModelScope，SHA256 27b80f2e…2595），板上用 b12425 的 llama-quantize 转 Q8_0/Q4_K_M
- 说明：最终微调 adapter 与真实照片不在本机克隆中；本表是原始模型（B0 权重）的板端部署与性能实测

## 单张图片冷启动（02 baseline，Q4_K_M，512²）

- 端到端 2.679 s，prefill 0.982 s，generate 1.533 s，生成 56 token
- 解析：ok；derived：{'decision': 'incomplete', 'missing_required': ['cable_bundle'], 'uncertain_required': [], 'answer_zh': '缺少线缆。'}

## 量化 × 像素档

| 量化 | 像素 | 视觉 token 上限 | 冷启动 p50 (n=3) | 冷 prefill p50 | 稳态 p50 / p95 (n=100) | 稳态 token/s | 解析 | 决策分布 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q4_k_m | 384² | 144 | 2.055 s | 0.404 s | 1.616 / 1.67 s | 38.63 | 100/100 | {'unknown': 34, 'incomplete': 33, 'complete': 33} |
| q4_k_m | 512² | 256 | 2.297 s | 0.556 s | 1.662 / 1.698 s | 37.88 | 100/100 | {'unknown': 34, 'incomplete': 33, 'complete': 33} |
| q4_k_m | 768² | 576 | 2.958 s | 1.226 s | 1.709 / 1.741 s | 37.48 | 100/100 | {'unknown': 34, 'incomplete': 33, 'complete': 33} |
| q8_0 | 384² | 144 | 2.492 s | 0.389 s | 1.983 / 2.083 s | 30.96 | 100/100 | {'unknown': 67, 'incomplete': 33} |
| q8_0 | 512² | 256 | 2.706 s | 0.534 s | 2.012 / 2.115 s | 30.64 | 100/100 | {'unknown': 67, 'incomplete': 33} |
| q8_0 | 768² | 576 | 3.221 s | 1.138 s | 2.046 / 2.073 s | 30.41 | 100/100 | {'unknown': 100} |

冷启动是每张冒烟图第一次出现时的 3 次请求；稳态是 100 次重复请求，会命中 llama-server 的 prompt cache，所以稳态 total_s 主要是生成耗时，不能当作全新照片的端到端延迟。

## 每档内存与功耗（每档 JSONL 写入窗口，不含模型加载）

| 量化 | 像素 | RAM 均值 / 峰值 (MB) | SWAP 峰值 (MB) | VDD_IN 均值 / 峰值 (mW) | GPU 频率均值 | 最高温度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q4_k_m | 384² | 5991 / 6146 | 2263 | 18145 / 19858 | 85.3% | 72.4 °C |
| q4_k_m | 512² | 6148 / 6251 | 2108 | 18363 / 20210 | 84.6% | 74.2 °C |
| q4_k_m | 768² | 6341 / 6488 | 2162 | 18246 / 20412 | 83.1% | 74.4 °C |
| q8_0 | 384² | 6773 / 6856 | 2249 | 18951 / 20412 | 86.6% | 74.3 °C |
| q8_0 | 512² | 6829 / 6963 | 2323 | 18960 / 21003 | 85.1% | 74.3 °C |
| q8_0 | 768² | 7057 / 7204 | 2325 | 18820 / 20451 | 84.0% | 74.3 °C |

## F16 语言权重

- F16（3.21 GiB）+ F16 mmproj（0.76 GiB）+ KV/compute 在 8GB 统一内存上被 OOM killer 终止。
- 内核记录：`Nov 25 17:21:43 yahboom kernel: Out of memory: Killed process 36705 (llama-server) total-vm:43220848kB, anon-rss:0kB, file-rss:615152kB, shmem-rss:0kB, UID:1000 pgtables:2272kB oom_score_adj:0`
- Q8_0（1.71 GiB）与 Q4_K_M（1.03 GiB）在同样条件下均完成 3 档像素的 100 次请求。

## 全流程内存与功耗（tegrastats，整段 9 档）

- RAM 已用：均值 6529 MB，峰值 7498 MB / 7620 MB
- SWAP 已用峰值：3422 MB
- VDD_IN：均值 17760 mW，峰值 21003 mW
- GPU 频率占比均值 78.7%，EMC 均值 未测%

## 演示界面（04，Q4_K_M，512²）

| 图片 | 墙钟 (s) | 推理 (s) | 决策 | 缺失/无法确认 |
| --- | ---: | ---: | --- | --- |
| bench_hammer_screwdriver.png | 2.395 | 1.871 | incomplete | 线缆 |
| bench_multimeter.png | 2.229 | 1.652 | complete | （清单内没有缺失项） |
| bench_empty.png | 2.145 | 1.627 | unknown | （清单内没有缺失项）；无法确认：螺丝刀、钳子、线缆、万用表、锤子 |

本机没有 `/dev/video*`，摄像头链路未测；演示使用「读取保存图片」模式，调用 Gradio `/infer` 接口完成同一推理路径。

## 未测项

- 板端真实照片准确率：`data/raw/session_001/` 与最终 adapter 都不在本机克隆中。
- 摄像头 FPS、现场演示视频：无摄像头设备。
- F16 的精度：语言权重在 8GB 上 OOM，未进入推理。
