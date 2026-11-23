# 实时摄像头点检（`scripts/live_check.py`）

一个独立窗口程序：打开摄像头显示实时画面，按 **空格** 把当前帧送给 Qwen3-VL 推理一次，按本次清单判断五类器材是否齐全。结论、每类状态和耗时直接画在画面上，同时打印到终端。

这是对 `qwen_vlm.app.app`（Gradio 网页）的补充：网页适合点按钮或上传图片，本程序适合“实时画面 + 键盘触发”的现场演示。两者都复用同一套冻结的 `Runner`、prompt 和 `derive`，不会改动 `contracts/`、`common/` 或 `inference/base.py`。

## 1. 快速开始

### Windows / 开发机（本机 Transformers）

本机需要 `torch`、`transformers`、`peft`（可选）、`opencv-python`、`Pillow`、`PyYAML`、`requests`。在仓库根目录：

```bash
python scripts/live_check.py --backend transformers
```

不写 `--model` 时，程序按以下顺序找模型：

1. 环境变量 `QWEN_VLM_MODEL_DIR`
2. `artifacts/base/Qwen3-VL-2B-Instruct`
3. 从仓库向上逐级查找 `PretrainedWeights/Qwen3-VL-2B-Instruct` 或 `Qwen3-VL-2B-Instruct`

例如本机的模型在 `Coursework/PretrainedWeights/Qwen3-VL-2B-Instruct`，会被自动找到。找不到时用 `--model` 指定：

```bash
python scripts/live_check.py --backend transformers --model "E:/大四课程/计算机视觉/Coursework/PretrainedWeights/Qwen3-VL-2B-Instruct"
```

有微调 adapter 时：

```bash
python scripts/live_check.py --backend transformers --adapter runs/M_final/adapter
```

> 仓库里没有微调后的 adapter 权重。本机只能先用基础模型；正式演示请用 Jetson 上的最终 GGUF（见下一节）。

### Jetson Orin Nano（llama-server）

最终部署用 GGUF。先在板子上启动 llama-server，方式二选一：

```bash
# 方式 A：用部署脚本同时起 llama-server 和 Gradio（Gradio 可以不用，留着也行）
cd ~/jetson_bundle_final
MODEL_TAG=final PIXELS=512 bash deploy/jetson/04_app.sh

# 方式 B：手动起服务，命令与部署说明一致
~/jetson_bundle_final/llama.cpp/build/bin/llama-server \
  -m ~/jetson_bundle_final/models/final-q4_k_m.gguf \
  --mmproj ~/jetson_bundle_final/models/mmproj-final-f16.gguf \
  --host 127.0.0.1 --port 8080 --jinja --temp 0 -c 4096 -ngl 99 -b 2048 -ub 2048 \
  --image-min-tokens 64 --image-max-tokens 256 --no-warmup --fit off
```

另开一个终端运行本程序（bundle 里没有 `scripts/` 时，把 `live_check.py` 和 `src/` 一并拷过去，或用项目源码目录）：

```bash
PYTHONPATH=src .venv/bin/python scripts/live_check.py \
  --backend llamacpp --server http://127.0.0.1:8080 --pixels 512 --camera /dev/video0
```

`--pixels` 必须和启动 llama-server 时的像素档一致（512 对应 `--image-max-tokens 256`），否则视觉 token 上限会对不上。

## 2. 按键

| 按键 | 作用 |
| --- | --- |
| `空格` / `Enter` | 拍下当前帧，送去模型推理一次 |
| `1`–`5` | 切换对应类别是否在本次清单内（螺丝刀、钳子、线缆、万用表、锤子） |
| `A` | 清单恢复为五类全选 |
| `R` | 清除上一次结论 |
| `Q` / `Esc` | 退出 |

推理在后台线程进行，画面继续刷新。推理期间再按空格会被忽略，界面提示“上一次推理还没结束”。

## 3. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--camera` | 摄像头编号，默认 `0`；Linux 也可写 `/dev/video0` |
| `--width` / `--height` / `--fps` | 请求的采集参数，默认 1280×720@30；相机不支持时会用实际分辨率 |
| `--backend` | `auto`（默认，先试 llama-server，失败则本机 Transformers）、`llamacpp`、`transformers`、`none`（只预览） |
| `--server` | llama-server 地址，默认 `http://127.0.0.1:8080` |
| `--pixels` | `384` / `512` / `768`，默认 512；显存紧张或要更快就用 384 |
| `--items` | 本次清单，中文名或类别 ID，空格分隔；默认五类全部。例：`--items 螺丝刀 钳子 万用表` |
| `--question` | 送给模型的中文问题，默认“按照本次实验清单，还缺少什么器材？” |
| `--adapter` | LoRA adapter 目录，仅 Transformers 后端 |
| `--device` / `--dtype` | 默认 `auto`；没有 CUDA 时自动用 CPU（会很慢） |
| `--save-dir` | 每次推理把画面和结果 JSON 存到这个目录 |
| `--font` | 中文字体文件；默认自动找微软雅黑 / Noto CJK |
| `--self-test` | 不加载模型，渲染一张自检画面到 `runs/live_demo/`，用来检查字体和布局 |

保存证据：

```bash
python scripts/live_check.py --backend llamacpp --save-dir runs/live_demo
```

每张图对应一个同名 JSON，字段包括 `captured_at`、`required_items`、`item_states`、`decision`、`missing_required`、`uncertain_required`、`answer_zh`、`timings` 和 `raw_text`，可直接用于演示录像或报告附录。

## 4. 结果怎么来的

- 模型只输出五类三状态 JSON（`present` / `absent` / `unknown`）。
- 是否齐全由 `src/qwen_vlm/common/derive.py` 确定性推导：
  - 清单内有 `absent` → 不齐全；
  - 否则有 `unknown` → 无法确认；
  - 否则全部 `present` → 齐全。
- 所以画面上的“结论”不是模型直接说的，而是程序按清单推导的，和评测口径一致。

## 5. 常见问题

**打不开摄像头。** 关掉占用摄像头的程序（浏览器、其他 OpenCV 窗口），Windows 试 `--camera 1`，Linux 用 `v4l2-ctl --list-devices` 找节点。Jetson 上当前用户不在 `video` 组时会没权限。

**窗口是英文。** 没找到中文字体。Windows 一般有微软雅黑；Jetson 装 `fonts-noto-cjk`，或用 `--font` 指定。终端输出始终是中文。

**没有窗口 / 只有 SSH。** 本程序用 OpenCV 本地窗口，需要显示器或 X 转发。远程访问请用 `qwen_vlm.app.app` 的网页模式。

**连不上 llama-server。** 先用 `curl http://127.0.0.1:8080/health` 确认服务在跑；确认 `--server`、端口和 `--pixels` 与启动参数一致。

**显存不够（CUDA out of memory）。** 先 `--pixels 384`；仍然不够就关掉其他占显存的程序。Jetson 上用 Q4_K_M，不要用 F16。

**本机没有 adapter，结果和报告里的数字对不上。** 本机基础模型只是开发调试用；报告和答辩的数字来自 `runs/` 里的折外评测，演示用 Jetson 上的最终 GGUF。

**按空格没反应。** 先点一下画面让窗口获得焦点；模型还在加载时界面顶部是黄色圆点，等变绿再试。
