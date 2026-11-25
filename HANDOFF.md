# 交接说明

课程：计算机视觉 Coursework 2（要求原文 `docs/coursework/cv_coursework2.docx`）。
项目：用 **Qwen3-VL-2B** 检查实验台上的器材是否齐全，包括螺丝刀、钳子、线缆、万用表、锤子。先做领域微调，再部署到 **Jetson Orin Nano 8GB**。
仓库：<https://github.com/zy-StarXH/qwen_vlm>

## 1. 现在做到哪了

| 部分 | 状态 |
| --- | --- |
| 环境、模型下载、接口定义 | 完成 |
| 训练 / 评测 / 转 GGUF / 演示界面 | 服务器第 1–5 步已完成，实际产物见 `STATUS.md` |
| Jetson Orin Nano 8GB | 本机重建原始权重 bundle，00–04 已验证（04 仅已保存图片模式；Q4/Q8 可运行，F16 OOM）；摄像头、最终 adapter、真实照片板端评测待补 |
| 真实照片 | `data/raw/session_001/` 共 50 张、9 个摆放组。状态与器材框已标注，用户已确认结束采集 |
| 原模型真实基线 | 512² 已评测 100 条 checklist，决策准确率 80/100，误报齐全 7/31，unknown 召回 2/18。见 `runs/B0_s001/eval/` |
| 正式三折 | B1、M、A1–A4 的 18/18 折已核对，每种方法 100 条折外预测。训练产物 `runs/cv3_folds/`，运行日志 `runs/logs/cv3_main.log` |
| 对比与模型选择 | 六种 LoRA 方法均为 2/31 误报齐全，按预定规则保留 M。详见 `runs/comparison/` 和 `EXPERIMENTS.md` |
| 鲁棒性 | 27/27 组已验收。M 在 `clean`、重度暗光、重度模糊分别判对 87/100、79/100、78/100。产物 `runs/robust/`，恢复命令见 `docs/NEXT_STEPS.md` |
| 最终 M | 全量 500 条完成 50 步。合并检查通过，已导出三档 GGUF。`runs/M_final/adapter/` 和 `release/jetson_bundle_final/` 均需在实例释放前备份；本机 clone 未包含这两项 |
| 报告与演示 | `reports/technical_report.md`、`reports/technical_report.zh.md`、`reports/demo_script.md` 和 `reports/defense_outline.md` 已完成。真实图片模式证据见 `runs/M_final/demo/` |

当前里程碑见 `STATUS.md`，运行命令和恢复位置见 `docs/NEXT_STEPS.md`。接手时先核对运行目录下的 manifest 和日志。

## 2. 本轮产物与下一步

B0 的服务器分辨率和量化扫描、框引导合成训练图、固定三折划分以及 `docs/NEXT_STEPS.md` 本轮第 1–5 步均已完成。正式模型选择沿用 M。

1. 释放实例前备份最终 adapter、部署包和项目报告。部署包中 `SHA256SUMS` 已覆盖 91 个文件。**本机 clone 不含服务器上的 `runs/M_final/adapter/`、`release/jetson_bundle_final/`、`data/raw/` 和 `vendor/`，需要从服务器备份恢复；本地 `release/jetson_bundle_base/` 是 2026-11-25 重建的原始权重 bundle。**
2. 板端已完成（2026-11-25）：在 Jetson Orin Nano 8GB 上用官方 Qwen3-VL-2B-Instruct F16 GGUF 重建了原始权重 bundle，llama.cpp b12425 CUDA 构建通过，`00`–`03` 跑通；Q4_K_M/Q8_0 完成 3 档像素各 100 次请求，F16 在 8 GB 上被 OOM killer 终止；`04` 的已保存图片模式经 Gradio `/infer` 验证。证据在 `runs/jetson/board_20261125/`（`summary.md` 为总表）。
3. 仍需完成：把最终 adapter 或 `release/jetson_bundle_final/` 以及 `data/raw/session_001/` 恢复到本机，才能做板端真实照片准确率和最终 M 的板端演示；摄像头当前没有 `/dev/video*`，现场演示与录像待接摄像头后补。

## 2.1 2026-11-25 板端实测（Jetson Orin Nano 8GB）

- 设备：Orin Nano 8GB Engineering Reference Developer Kit Super，L4T R36.4.3，CUDA 12.6.11，MAXN_SUPER；7.4 GiB RAM + 11 GiB swap。
- 模型来源：ModelScope 官方 `Qwen/Qwen3-VL-2B-Instruct-GGUF` F16（SHA256 `27b80f2e…2595`），板上用 b12425 转 Q8_0/Q4_K_M；mmproj F16 的 SHA256 为 `c3d5afbe…aea5`。
- `00`–`01` PASS：llama.cpp b12425 CUDA sm_87 构建成功；`02` Q4_K_M 512² 冷启动 2.68 s（prefill 0.98 s，生成 56 token）。
- Q4_K_M：冷启动 p50 2.06 / 2.30 / 2.96 s（384²/512²/768²）；稳态 p50 1.62 / 1.66 / 1.71 s；生成 38.6 / 37.9 / 37.5 token/s。
- Q8_0：冷启动 p50 2.49 / 2.71 / 3.22 s；稳态 p50 1.98 / 2.01 / 2.05 s；生成 31.0 / 30.6 / 30.4 token/s。
- 资源：Q4 峰值 6.15–6.49 GB RAM、19.9–20.4 W；Q8 6.86–7.20 GB、20.4–21.0 W（含桌面与 agent）。F16 语言权重被 OOM killer 终止。
- 未测：板端真实照片准确率（缺 `data/raw/session_001/` 与最终 adapter）、摄像头 FPS/视频（无 `/dev/video*`）。
- 复现：重建的 bundle 在 `release/jetson_bundle_base/`（模型约 6.8 GiB，总计约 8 GB，git 忽略）；来源、哈希和命令见 `runs/jetson/board_20261125/README.md` 与 `models_manifest.json`。

## 3. 怎么开始

**在原服务器上接手**：所有东西都在 `/root/gpufree-data/qwen_vlm`，每次先执行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
envs/qwen_vlm/bin/python -m pytest -q tests
```

**在新机器上从 GitHub 开始**：仓库里只有代码和文档，下面这些要自己准备：

| 缺的东西 | 怎么补 |
| --- | --- |
| conda 环境 | `conda env create -p envs/qwen_vlm -f env/environment.yml && envs/qwen_vlm/bin/pip install -e .` |
| 基础模型（4.3GB） | `python scripts/download_base_model.py`（会逐个文件校验哈希） |
| llama.cpp 源码 | 下载 tag `b12425` 的源码，解压到 `vendor/llama.cpp`（见 `deploy/jetson/LLAMACPP_VERSION.md`），再运行 `bash scripts/export_build_llamacpp_server.sh` |
| 真实照片 | 不进 git，找原作者要 `session_001.zip`，解压到 `data/raw/` |
| GGUF 和模型权重 | 用 `scripts/export_gguf.sh` 重新生成；或者从 ModelScope 下官方 `Qwen3VL-2B-Instruct-F16.gguf` + `mmproj-Qwen3VL-2B-Instruct-F16.gguf`，再用板上的 `llama-quantize` 转 Q8_0/Q4_K_M（2026-11-25 板端实测走的就是这条路） |
| 最终 LoRA adapter | 仓库和 clone 里都没有，只在服务器备份 `runs/M_final/adapter/`；没有它就无法复现最终 M 的板端结果 |

`env/activate.sh` 里写死了 `/root/gpufree-data/qwen_vlm`，在别的路径使用时改第 3 行。

## 4. 必须知道的规则

- **不要改冻结的接口**：`contracts/`、`src/qwen_vlm/common/`、`src/qwen_vlm/inference/base.py`。改了会让前后实验没法比较；确实要改，就把版本号加一并记下来。
- **不编造结果**：合成图（`data/smoke/`）和冒烟运行（`runs/*smoke*`）的任务质量数字一律不能写进报告；板端冒烟图只允许写延迟/资源，并须注明不是准确率。没跑的实验就写"待测"。
- **测试集冻结**：划分好之后，选模型和调参只能看验证集。
- **同一摆放的照片不能拆开**：程序按"摆放编号"划分数据，标注时编号一定要填对。
- **磁盘**：项目盘已扩容到 99 GiB（2026-11-21 剩约 59 GiB），合并模型和 GGUF 直接放在 `artifacts/` 下。训练 checkpoint 只保留一个，单折预测验证后保留 adapter。`/root/gpufree-data` 在释放实例时不保存，重要产物要提醒用户下载备份。
- **网络**：服务器外网很慢，能用的下载方法写在 `env/activate.sh` 的注释里。推送 GitHub 要开本地代理，或者不走代理直连。

## 5. 文件导航

| 想做什么 | 看哪里 |
| --- | --- |
| 所有命令 | `README.md` |
| 课程要求逐条对应 | `docs/coursework_requirements.md` |
| 怎么拍、怎么标 | `docs/data_collection.md`、`docs/annotation_guide.md` |
| 实验设计（B0/B1/M/A1–A3、优化矩阵） | `EXPERIMENTS.md` |
| 指标定义 | `contracts/metrics.md` |
| 模型原理（写报告用） | `docs/model_principles.md` |
| 板端部署 | `deploy/jetson/README.md` |
| 风险和已知问题 | `reports/readiness.md` 第 4 节 |
| git 怎么用 | `docs/git_workflow.md` |
| 快速测试一张图 | `python scripts/check_image.py <图片> --items 螺丝刀 钳子` |

## 6. 还没确认的事

- 截止时间、报告篇幅与排版格式、提交方式。（已定：板子用 Jetson Orin Nano 8GB，不再向老师确认；报告英文为正式版，另附中文翻译供用户校对。）
- 拍摄时相机用的是自动曝光，以后最好改成手动锁定。
