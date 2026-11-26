# 就绪报告

2026-11-21 已核对正式三折的 18 份 50 步 manifest 和 B1、M、A1–A4 各 100 条真实折外预测。训练峰值显存分配为 5.396–5.411 GiB（L40）。鲁棒性九档各校验 0 错误，B0、B1、M 共 2700 条预测成功解析。M 在重度暗光和模糊分别少判对 8、9 条。恢复命令见 `docs/NEXT_STEPS.md`。2026-11-25 用原始权重完成板端构建、Q4/Q8 扫描和图片模式验证；2026-11-26 恢复最终 Q4_K_M/mmproj/adapter 和 50 张真实照片，完成最终 M 的三档计时、训练集部署检查和四张演示图。摄像头仍待完成。

2026-11-20 更新。50 张真实照片均已标注，含 9 个摆放组和 100 条 checklist。L40 上已完成 B0 三档分辨率和三档量化评测，见 `runs/comparison/`。器材框增强和固定三折划分已用全部数据生成和校验，见 `data/splits/cv3/`。正式训练和板端实测的进度见 `STATUS.md` 与 `docs/NEXT_STEPS.md`。

正式对比显示，六种 LoRA 方法的误报齐全率均为 2/31，B0 为 7/31，默认部署用 M。M 的决策准确率为 86/100，真实遮挡 unknown 召回为 1/14。消融、样本不足及逐样本复核见 `EXPERIMENTS.md`。

最终 M 的 500 条训练样本通过校验并完成 50 步。代表性样本复核了合并前后和重新加载的文本一致性。L40 用最终导出的 Q4_K_M 和 Q8_0 各运行 100 条训练照片清单，输出全部可解析。`release/jetson_bundle_final/` 的 91 项 SHA256 全量通过。证据见 `runs/M_final/`。

最终 Q4_K_M 的 Gradio 图片模式已在 L40 验证，四条请求均可解析。真实齐全图 `v001` 多判线缆 unknown，清晰对照 `v040` 判齐全。缺万用表 `v014` 判缺失，手部遮挡 `v041` 判无法确认并多报线缆 unknown。记录见 `runs/M_final/demo/image_mode_check.json`。双语报告、演示脚本和答辩提纲已交付。2026-11-25 板端（Jetson Orin Nano 8GB）用原始权重重建 bundle，`00`–`03` 跑通，Q4/Q8 各三档像素全部解析，F16 OOM。2026-11-26 恢复最终 Q4_K_M bundle 与 50 张照片，最终 M 完成三档计时、100 条真实照片部署检查（77/100 决策、2/31 误报齐全、16/18 unknown 召回，与 L40 一致）和四张演示图。摄像头、录像和最终 Q8/F16 未完成。证据见 `runs/jetson/final_20261126/`。

以下保留了 2026-11-16 的开发验证记录，冒烟指标仅用于复核链路。正式实验的真实数据结果见 `STATUS.md` 和 `EXPERIMENTS.md`。

## 1. 已验证

| 项目 | 证据 | 要点 |
| --- | --- | --- |
| 项目目录与路径规则 | `env/activate.sh` | 缓存、临时文件、pip/conda 包都在项目内。修复了全局 `HF_HUB_CACHE` 覆盖问题和系统 cuDNN 9.8 冲突 |
| conda 环境 | `env/environment.lock.yml`，`reports/environment_server.md` | `envs/qwen_vlm`：py3.12，torch 2.11 cu130，transformers 5.5.4，peft 0.21.3 |
| 基础模型固定版本 | `artifacts/base/manifest.json` | revision `89644892…`，11 个文件逐一与 HF 记录的哈希一致 |
| 冻结接口 | `contracts/`，`src/qwen_vlm/common/`，`src/qwen_vlm/inference/base.py` | 类别、schema、推导规则、Runner、指标定义、训练混合规则 |
| 单元测试 | `tests/` | `91 passed`。包括手算指标核对、标签屏蔽、LoRA 不匹配视觉模块、数据泄漏检测 |
| B0 单图推理（Transformers） | `runs/smoke_b0/` | JSON 可解析；图像确实影响 logits（与纯灰图相比最大差 5.19）；三档像素对应 130/234/540 个视觉 token；显存峰值 4.2GiB |
| 数据流水线 | `data/smoke/synthetic/` | CSV 转 JSONL、校验、按场景组划分、泄漏检查全部跑通（合成 24 图 / 72 条） |
| LoRA 训练链路 | `runs/train_smoke/` | 只有语言模型 196 个投影可训练（17.4M，占 0.81%）；loss 从 0.567 降到 0.002（过拟合 4 条）；显存约 5.5GiB；adapter 保存、加载、续训正常 |
| LoRA 合并 | `runs/train_smoke/merge_check.json` | 合并前后贪心文本一致；logits 最大差 3.75（bf16 舍入，见风险 3） |
| GGUF 转换 | `artifacts/quantized/base/manifest.json` | llama.cpp `b12425`；语言 F16 3.3G / Q8_0 1.8G / Q4_K_M 1.1G；mmproj（视觉+DeepStack）F16 782M |
| 像素档跨后端一致 | `runs/deploy_smoke/resolution_check.json` | llama.cpp 与 Transformers 三档视觉 token 都是 130/234/540，说明分辨率优化在两个后端是同一个变量 |
| 端到端（两个后端） | `runs/integration_smoke/` | 预测、评测、出表在 Transformers 和 GGUF（F16/Q8_0/Q4_K_M）上都跑通；新增 `scripts/predict_llamacpp.py`（不依赖 torch，可在板上使用） |
| 演示界面（图片模式） | `runs/deploy_smoke/app_smoke.json` | Gradio + llama-server(Q4) 完整调用成功；单请求锁有单测 |
| Jetson 侧导入链 | 本报告 | 导入 `llamacpp_runner` 与 `app` 时不会引入 torch/transformers/peft |
| Jetson 板端构建与推理 | `runs/jetson/board_20261125/` | L4T R36.4.3、CUDA 12.6.11、MAXN_SUPER；b12425 CUDA sm_87 构建 PASS；Q4_K_M 512² 冷启动 2.68 s |
| 板端量化 × 像素扫描 | `runs/jetson/board_20261125/summary.md` | Q4_K_M/Q8_0 × 384²/512²/768² 共 6 档，每档 100 次请求全部解析；F16 被 OOM killer 终止 |
| 板端 Gradio 图片模式 | `runs/jetson/board_20261125/demo/` | 三张冒烟图经 `/infer` 全部解析；无摄像头设备 |
| 最终 M 板端运行 | `runs/jetson/final_20261126/` | final Q4_K_M/mmproj/adapter 与服务器 bundle 逐字节一致；Q4 三档计时与 02/04 跑通 |
| 最终 M 真实照片部署检查 | `runs/jetson/final_20261126/summary.md` | 100 条训练照片：77/100 决策、2/31 误报齐全、16/18 unknown 召回；与 L40 决策一致 96/100。不是泛化指标 |
| 最终 M 板端演示 | `runs/jetson/final_20261126/demo/` | v001/v014/v040/v041 决策与 L40 一致；板端推理 3.16–3.57 s |

**一个有用的早期观察**（合成图，只用于定位差异）：在 15 条合成 val 上与 Transformers bf16 逐条比较，GGUF Q8_0 完全一致（15/15），F16 为 14/15，Q4_K_M 只有 8/15（类别级 36/50）。说明 Q4 对输出的影响不可忽略，量化实验必须在真实测试集上测任务指标，不能只报速度。（`runs/integration_smoke/agreement_vs_bf16.txt`）

## 2. 已准备但未验证

| 项目 | 位置 | 缺什么才能验证 |
| --- | --- | --- |
| 最终 Q8_0/F16 板端对照 | `final-q8_0.gguf`、`final-f16.gguf`（未上传） | 从服务器备份恢复；F16 在 8 GB 上已知 OOM，Q8 可用于高精度对照 |
| 摄像头演示 | `src/qwen_vlm/app/app.py`，`04_app.sh` | 本机没有 `/dev/video*`，接摄像头后测 FPS 和录像 |
| 合并后的独立评测 | `runs/M_final/merge_check.json` | 已检查代表性样本文本及服务器导出链路。最终 adapter 使用全量样本；板端部署检查已完成（见 `runs/jetson/final_20261126/`），独立光照验证待测 |
| 资源与延迟统计 | `src/qwen_vlm/evaluation/latency.py`，`tegrastats.py` | 原始权重 Q4/Q8 与最终 M Q4 板端日志已解析；相机 FPS 仍缺 |

## 3. 等待用户材料（按优先级）

1. 接上 USB 摄像头（当前无 `/dev/video*`），补现场演示、相机 FPS 和录像。
2. 如需最终 Q8/F16 的板端对照，从服务器备份恢复 `final-q8_0.gguf`/`final-f16.gguf`；F16 在 8 GB 上已知 OOM。
3. 核实截止时间、报告篇幅与排版格式、提交渠道。（板子已定为 Jetson Orin Nano 8GB；报告英文为正式版，另附中文翻译。）

## 4. 风险

1. **服务器网络很慢**（直连约 30-50KB/s）。可用路线已记录在 `env/activate.sh`：ModelScope 走代理、阿里云 PyPI 直连、gh-proxy 访问 GitHub。新增大依赖要预留时间。
2. **磁盘**：项目盘已扩容到 99 GiB（2026-11-21 剩约 59 GiB），每次合并模型约 4.3GB、每组 GGUF 约 7GB，不再紧张。剩下的风险是 `/root/gpufree-data` 在释放实例时不保存，最终 adapter、GGUF 和 bundle 需要另外备份。
3. **合并误差**：bf16 合并在 logits 上有可见误差，冒烟样本的贪心输出没有变化。正式模型要以 val 集任务分数一致作为合并通过的标准。
4. **板端内存**：2026-11-25 原始权重 Q4_K_M/Q8_0 都能在 8 GB 上跑完三档，F16 被 OOM killer 终止；2026-11-26 最终 M Q4_K_M 三档跑完，峰值 RAM 7.22–7.31 GB（含桌面与 agent）。最终 Q8/F16 未上传；板端高精度参照仍可用 2026-11-25 原始权重的 Q8_0。
5. **Jetson 环境**：脚本假设 JetPack 6 / Python 3.10，需要 `python3-venv`，`01` 需要网络安装 gradio 等。如果板子没网，需要另外准备离线 wheel。
6. **进程内存**：tegrastats 只有整机内存。如果需要进程级内存，要在 bench 脚本里额外采样 `/proc/<pid>/status`（后续可以补）。
7. **unknown 的边界**：B0 512² 在真实图上只召回 2/18 个 unknown 标签。合成遮挡针对这一问题补充监督。
