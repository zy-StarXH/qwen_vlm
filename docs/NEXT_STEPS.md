# 下一步执行计划（交给接手的模型/同学）

更新：2026-11-25。项目概况和接手规则见 `HANDOFF.md`。本文件记录实验计划、已验证的产物和恢复命令。

## 本轮要做的事（2026-11-21 交接，先读这一节）

2026-11-21 已完成本轮第 1–5 步，含正式三折、对比复核、27 组鲁棒性评测、最终 M 模型导出、服务器图片演示和报告交付。2026-11-25 在 Jetson Orin Nano 8GB 上用本机重建的原始权重 bundle 完成第 6 步的 00–04（04 为图片模式演示）；最终 adapter、真实照片板端评测和摄像头仍待补。`release/jetson_bundle_final/` 与 `runs/M_final/adapter/` 请在实例释放前下载备份。

**Jetson 板端第 6 步已于 2026-11-25 部分完成**（00–04 与图片模式演示）；最终 adapter、真实照片板端评测和摄像头仍待补，见本节后面的第 6 步。课程要求与证据的对应关系见 `docs/coursework_requirements.md`。第 3、5 步是为了补上课程要求里的缺口：鲁棒性评估、技术报告、演示和答辩材料。

按顺序做。每步做完跑 `pytest -q tests` 并单独 `git commit`。第 3 步中 B0 的部分不依赖训练结果，显存也够（训练约占 5.5 GiB），可以和第 1 步同时跑。

### 第 1 步：跑正式三折对比（任务 C 第 3 点）

**已完成。** 正式任务从 15:24 至 17:30，耗时约 2 小时 6 分钟。18 折各完成 50 步，adapter 均可用于推理，600 条汇总预测全部解析成功。每折测试集为 50、34、16 条，按同一金标准顺序合并。训练峰值分配为 5.396–5.411 GiB，峰值保留为 5.566–5.662 GiB，设备为服务器 L40。训练、数据和冻结接口均与 `a9a088f` 一致。manifest 的 `dirty` 标记包含运行产物与进度文档。全量测试 111 通过，复核日志见 `runs/logs/cv3_audit.log`。

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
P=envs/qwen_vlm/bin/python
mkdir -p runs/logs
nohup $P -u scripts/run_cv.py --max-steps 50 >> runs/logs/cv3_main.log 2>&1 < /dev/null &
printf '%s\n' "$!" > runs/logs/cv3_main.pid.log
```

- 共 6 种方法 × 3 折 = 18 次训练加预测，脚本串行执行。M 的 30 步试跑用了约 3.3 分钟，据此估计每次 50 步训练加预测约 6–8 分钟，整轮约 2–3 小时。这是估算，以日志为准。
- 中断后用原命令重跑，已完成的 adapter 和预测会复用。不要改步数、配置或划分；必须改时换新的 `--runs-root`，不要覆盖旧结果。
- 从短生命周期的命令执行工具启动时，后台子进程曾随 shell 退出。工具会话内启动后追加 `wait "$(cat runs/logs/cv3_main.pid.log)"`，保留 `nohup` 和持续会话。先核对是否已有进程，恢复记录见 `runs/logs/cv3_recovery.log`。
- 每折结束后脚本会自动删除 `checkpoints/`，只留 `adapter/`。

验收：

- `runs/{B1,M,A1,A2,A3,A4}_cv3/predictions.jsonl` 各 100 行，sample_id 与 `data/annotations/real_v1.jsonl` 一一对应，且都有 `eval/metrics.json`。
- 每个 `runs/cv3_folds/<ID>_fold<k>/run_manifest.json` 中 `steps_completed` 为 50，`error` 为空。

### 第 2 步：汇总对比表并写结论（任务 D）

**已完成。** 12 组运行重新出表，三折方法列全部有实测值。6 种 LoRA 方法的误报齐全率均为 2/31，B0 为 7/31。M 决策准确率 86/100、unknown 召回 1/18。没有方法满足严格更低的误报齐全率，按预定规则保留 M。逐样本结果与成对区间见 `runs/comparison/cv_case_review.txt`，结论见 `EXPERIMENTS.md`。

```bash
$P scripts/eval_compare.py --runs-root runs --out runs/comparison
```

对比表中 B1、M、A1–A4 列应全部有数。然后在 `STATUS.md`、`EXPERIMENTS.md`、`reports/readiness.md` 中用大白话回答下面四个问题，只写实测数字：

1. 微调有没有用：B1、M 相比 B0 的误报齐全率（B0 为 7/31）和决策准确率（B0 为 80/100）。
2. 合成图有没有用：比较 M 和 A4（mix 相同，A4 不加合成图）。
3. 各组件的贡献：M 对比 A1（去掉成对辅助）、A2（去掉困难负样本加权）、A3（去掉 unknown 样本）。
4. B0 的老毛病改了没有：逐张对照 `predictions.jsonl`，检查第 14、22、27、32 张“有线缆没万用表却判万用表在”、螺丝刀被判 unknown、遮挡时不说 unknown 这三类错误。

只有 9 个摆放组，置信区间一律标“样本不足”；差距落在区间内时写“差异不显著”，不要写成“提升”。必须披露的局限见任务 D 第 3 点。

### 第 3 步：鲁棒性实验（课程要求 (4) “robustness under different conditions”）

**已完成。** 已生成 450 张 JPEG 和 900 条评测样本，九档校验均为 0 错误。B0、B1、M 共 27 组各 100 条预测全部成功解析，B1 和 M 的预测与各自的折外 adapter 校验一致。B0 推理与训练并行约 25 分钟，B1 与 M 串行折外推理共 67 分钟。M 在 `clean`、重度暗光、重度模糊的决策准确率分别为 87/100、79/100、78/100。对比表、真实条件子集及目视审核图见 `EXPERIMENTS.md`，完整生成清单见 `runs/robust/dataset_manifest.json`。

复现数据用 `$P scripts/data_make_robust.py --seed 0`。加载 `env/activate.sh` 后执行下列命令可恢复 B1 与 M，已验证的预测将复用。B0 使用 `--models B0`。

```bash
P=envs/qwen_vlm/bin/python
nohup $P -Bu scripts/run_robust.py --models B1 M >> runs/logs/robust_cv3.log 2>&1 < /dev/null &
printf '%s\n' "$!" > runs/logs/robust_cv3.pid.log
```

真实数据里只有遮挡（10 条）和有手（6 条）两种条件，没有光照变化，用户也不再拍新照片。因此对真实测试照片做人工干扰，只做推理、不训练，比较各模型在干扰加重时掉多少。报告里必须写明这是模拟干扰，不是真实拍摄。

1. **生成干扰图**：新增 `scripts/data_make_robust.py`（实现放在 `src/qwen_vlm/data/robust.py`，单测 `tests/test_data_robust.py`）。输入 `real_v1.jsonl`，固定随机种子，输出到 `data/robust/v1/`，并把 `data/robust/` 加进 `.gitignore`。
   - 条件：暗（亮度 ×0.5、×0.3）、过曝（×1.5、×2.0，截断到 255）、高斯模糊（σ=2、4）、高斯噪声（σ=15、30），共 8 档。另加一档 `clean`：不做干扰、只走同一保存流程，作为对照，避免把 PNG 转 JPEG 的影响算到干扰头上。
   - 图片保持原分辨率，存为 JPEG（q=95）。每档单独一个 `samples.jsonl`，标签不变，`session_id`、`scene_group_id` 不变。`condition_tags` 追加如 `robust_dark_l1`。sample_id 和 `pair_group_id` 加上档位后缀，命名规则参照 `augment.py`（编号里不能有下划线）。
   - 先拼缩略图抽查：最重的档里人眼应仍能看清器材，否则“标签不变”不成立，需减轻该档。每档都要过 `scripts/data_validate.py`，结果 0 错误。
2. **跑预测**：新增 `scripts/run_robust.py`。B0 不加 adapter，按 512² 跑全部 9 档。B1 和 M 必须**折外**预测：每张图用它所在测试折的 adapter（`runs/cv3_folds/<ID>_fold<k>/adapter`，分组见 `FIXED_CV3_TEST_GROUPS`），三折结果拼成 100 条后，再用 `eval_predictions.py` 评测。输出到 `runs/robust/<ID>_<条件>/`。每个模型 900 条预测，预计 20–30 分钟，三个模型约 1.5 小时（估算）。
3. **出表**：B0、B1、M 在每档上的决策准确率、误报齐全率、unknown 召回，以及相对 `clean` 下降了多少。真实条件（遮挡、有手）的结果从第 2 步各 `metrics.json` 的 `by_condition` 中读取，放在同一章并单独标为“真实条件”。

验收：9 档 × 3 个模型都有 100 条预测和 `eval/metrics.json`；单测通过；表和抽查缩略图写入 `EXPERIMENTS.md`。

### 第 4 步：训练最终模型并导出（任务 E 第 1–3 点）

**选模型规则（看结果前定好，不要事后改）**：默认部署 M。只有当另一种方法的误报齐全率比 M 低、且决策准确率不比 M 差时，才停下来把对比交给用户决定是否换。不在测试集上再调任何参数。

**已完成。** 全量训练 JSONL 为 500 条、250 张、0 错误，SHA256 `f36a53d6284a840e691f3ae3d1e62fcc513cfc19243aecf7e84d74a5c1d3671a`。M 的 50 步训练正常结束，日志 `runs/logs/train_M_final.log`。代表性样本合并前、合并后与重新加载的贪心文本一致。最终 Q4_K_M、Q8_0 在 L40 分别完成 100 条训练照片导出检查并全部解析成功。合并、导出和打包日志为 `runs/logs/merge_M_final.log`、`runs/logs/export_M_final.log`、`runs/logs/bundle_M_final.log`。

交付目录 `release/jetson_bundle_final/` 已核对 91 个文件的 SHA256。包内含三档语言 GGUF、F16 mmproj、最终 adapter、合并证据、部署说明与项目源码快照 `31f44b7`；归档校验表在 `runs/M_final/bundle_SHA256SUMS`。下载时同时备份 `runs/M_final/adapter/` 与整个 bundle。重跑时先核对 `runs/M_final/run_manifest.json` 和 adapter，完整训练结果可直接从合并继续。

```bash
# 全量训练数据：100 条真实 + 400 条合成，已验证 0 错误
cat data/annotations/real_v1.jsonl data/aug/v1/samples.jsonl > data/splits/full_v1.jsonl
$P scripts/data_validate.py data/splits/full_v1.jsonl
nohup $P -Bu scripts/train.py --config configs/train/M.yaml --train-jsonl data/splits/full_v1.jsonl \
   --no-validation --max-steps 50 --output-dir runs/M_final >> runs/logs/train_M_final.log 2>&1 < /dev/null &
printf '%s\n' "$!" > runs/logs/train_M_final.pid.log
```

训练正常结束并检查 adapter 后运行以下命令。

```bash
rm -rf runs/M_final/checkpoints

# 合并与导出都放在项目盘，沿用 base 的目录约定（这些目录已被 git 忽略）
$P scripts/merge_lora.py --adapter runs/M_final/adapter --sample-jsonl data/annotations/real_v1.jsonl \
   --merged-dir artifacts/merged/final --merge-check runs/M_final/merge_check.json
$P scripts/export_gguf.py artifacts/merged/final artifacts/quantized/final final
INCLUDE_QUANTS=q4_k_m,q8_0,f16 scripts/make_jetson_bundle.sh artifacts/quantized/final release/jetson_bundle_final
```

- 文件名必须带 `final` 标签（生成 `final-q4_k_m.gguf`、`mmproj-final-f16.gguf`），Jetson 脚本靠 `MODEL_TAG=final` 找模型。
- 空间：项目盘已扩容到约 59 GiB 可用。合并模型约 4.3 GiB，全套 GGUF 约 6.8 GiB，空间足够。合并权重可以保留，以后重新量化时不必再合并。打包脚本用硬链接引用 `artifacts/quantized/final/` 中的模型，所以 bundle 本身很小。
- `/root/gpufree-data` 在关机后数据还在，但**释放实例时不保存**。最终 bundle 和 adapter 做好后，提醒用户下载备份。
- 可以用 `scripts/predict_llamacpp.py` 让最终模型的 Q4_K_M 和 Q8_0 在真实 100 条上跑一遍，确认导出没坏。最终模型训练时见过这些图，这组数只能当导出检查，**不能当泛化指标写进报告**；泛化指标只用第 2 步的折外结果。

- 交付物要求“源码、模型文件、部署说明”齐全：把 `runs/M_final/adapter/` 和 `merge_check.json` 复制进 bundle（例如 `models/adapter/`），再按脚本已有的 `SHA256SUMS` 格式重新计算校验和。
- 各量化档的 GGUF 文件大小与设备无关，可以直接写进报告的资源部分。L40 上测到的延迟和显存**不能**当作 Jetson 的数字。

验收：`runs/M_final/merge_check.json` 显示合并前后输出一致；`release/jetson_bundle_final/` 内有 `final-q4_k_m.gguf`、`mmproj-final-f16.gguf`、adapter 和更新后的 `SHA256SUMS`。

### 第 5 步：技术报告初稿、演示与答辩材料（课程 Deliverables）

**已完成。** 英文正式版 `reports/technical_report.md`、中文逐段校对版 `reports/technical_report.zh.md`、视频脚本 `reports/demo_script.md` 和五分钟答辩提纲 `reports/defense_outline.md` 均已交付。最终 Q4_K_M 以 512² 在 L40 的 Gradio 图片模式验证了三类场景。`v001` 齐全但线缆输出为 unknown，补测的清晰齐全图 `v040` 正确判齐全。`v014` 正确判缺万用表，`v041` 正确判无法确认，但多报了线缆 unknown。两版报告保留四张原始输出截图。证据见 `runs/M_final/demo/image_mode_check.json`。

复现时先加载 `env/activate.sh`，启动下列本机服务。另一终端加载相同环境并启动 Gradio，在“读取保存图片”模式勾选五项即可。

```bash
vendor/llama.cpp/build/bin/llama-server \
  -m artifacts/quantized/final/final-q4_k_m.gguf \
  --mmproj artifacts/quantized/final/mmproj-final-f16.gguf \
  --host 127.0.0.1 --port 8097 --jinja --temp 0 -c 4096 -ngl 99 -b 2048 -ub 2048 \
  --image-min-tokens 64 --image-max-tokens 256 --no-warmup --fit off
```

```bash
envs/qwen_vlm/bin/python -m qwen_vlm.app.app --server http://127.0.0.1:8097 --pixels 512
```

1. **技术报告**：英文正式版 `reports/technical_report.md`，中文翻译 `reports/technical_report.zh.md` 供用户校对。
   - 两版章节、表格、数字、图一一对应。先写英文，再逐段翻译成中文；中文版开头注明“以英文版为准”。英文版改动后要同步修改中文版。
   - 术语首次出现时，中文版保留英文原词，例如“误报齐全率（false-complete rate）”，方便用户对照。
   - 设备统一写 “Jetson Orin Nano 8GB”，在部署章说明它属于 Jetson Nano 系列。
   - 篇幅和排版课程原文没有规定，先用 Markdown，之后按用户要求转换格式。
   
   章节按课程要求排：
   - 系统设计（任务、器材清单、整体流程）；
   - 模型原理（在 `docs/model_principles.md` 基础上补全）；
   - 改进方法（LoRA、成对辅助、困难负样本、unknown 监督、框引导合成数据）；
   - 数据集（data card，模板 `docs/data_card_template.md`）；
   - 部署（llama.cpp、GGUF、Jetson 脚本）；
   - 优化（量化、分辨率）；
   - 实验评估（对比、消融、鲁棒性、速度与资源）；
   - 局限与改进；
   - 结论。
   
   实验章参照 `reports/templates/`。数字只从 `runs/` 的 `metrics.json` 取。板端章节写已实测的原始权重构建、延迟、吞吐、内存和功耗，并注明只用于性能、不是任务质量；最终 M、真实照片准确率和摄像头仍写“待测”，不要用服务器数字顶替。
2. **服务器上验证演示程序**：用最终模型的 Q4_K_M 在服务器上以图片模式启动 `src/qwen_vlm/app/app.py`，跑三类典型样例：器材齐全、有线缆但没万用表、遮挡。截图放进报告，作为板端演示前的备份证据。
3. **演示视频脚本** `reports/demo_script.md`：逐个镜头写清楚。板子和摄像头的整体布置；齐全场景；拿走万用表但留下线缆；手遮住器材时模型回答 unknown；画面上显示延迟。拍摄要等摄像头到位。
4. **答辩提纲** `reports/defense_outline.md`：用 5 分钟讲清问题、方法、结果。列出可能被问到的问题和答案，例如：为什么选 VLM 而不是 YOLO、为什么用 Orin Nano 8GB 跑 2B 的 VLM、数据只有 50 张怎么保证结论可信、合成图会不会造成泄漏、量化带来的精度和速度取舍。

验收：两版报告、演示脚本与答辩提纲共四份文档已提交。章节、段落、表格、数字与截图逐项核对。`STATUS.md` 的 P6 行已更新。

### 第 6 步：Jetson 实测（2026-11-25 部分完成）

板子已到手。因为 clone 没有服务器上的 `release/`、`runs/M_final/adapter/` 和 `data/raw/`，本轮用官方 Qwen3-VL-2B-Instruct F16 GGUF 在本机重建了原始权重 bundle，完成：

- `00_probe`、`01_build_llamacpp`（b12425，CUDA sm_87）、`02_run_baseline`（Q4_K_M 512²）PASS。
- `03_bench`：Q4_K_M/Q8_0 × 384²/512²/768² 共 6 档，每档 5 次预热 + 100 次正式请求，全部解析；F16 在 8 GB 上被 OOM killer 终止。
- `04_app`：Gradio 已保存图片模式经 `/infer` 接口验证三张冒烟图，全部解析。
- 数字、日志、模型哈希：`runs/jetson/board_20261125/`（先看 `summary.md`）。

仍需完成，依赖用户备份或硬件：

1. 恢复 `runs/M_final/adapter/` 或 `release/jetson_bundle_final/`，用 `MODEL_TAG=final` 重跑 `02`–`04`，得到最终 M 的板端输出。
2. 恢复 `data/raw/session_001/` 后，在板上按 `deploy/jetson/README.md` 跑 `predict_llamacpp.py`，回传预测 JSONL 与 `tegrastats`，评测真实照片。最终模型训练时见过这些图，只能当部署检查，不能当泛化指标。
3. 接上摄像头后跑 `04_app.sh`，录现场演示视频并补相机 FPS。
4. 报告板端章节已填入原始权重的构建、延迟、吞吐、内存和功耗；任务质量、最终 M、摄像头仍标待测。

板子型号已定：Jetson Orin Nano 8GB。截止时间、报告篇幅与排版格式仍待用户告知。

## 0. 当前状态（已完成，不要重做）

- 环境、基础模型、冻结接口、训练/评测/GGUF/演示/Jetson 脚本全部就绪。
- 真实数据：`data/raw/session_001/` 50 张（1920×1080），**用户决定不再拍新数据**。
  - 状态标注：`data/annotations/s001.csv` → `data/annotations/real_v1.jsonl`（100 条 checklist，0 错误）。
  - 器材框（已人工复核，可当真值）：`data/annotations/annotations.json`，格式见 `data/annotations/README.md`（`cable` = `cable_bundle`，坐标 xyxy 像素）。
  - 摆放分组（按框位置确定，跨组无近重复）：第 1–6 张 l001–l006 各一组，7–8 张 l007，9–25 张 l008，26–50 张 l009。
- B0（原模型，Transformers bf16，512²）已在全部 50 张上评测：`runs/B0_s001/eval/metrics.md`。
  决策准确率 80/100，**误报齐全率 7/31**，unknown 召回 2/18。典型错误包括第 14、22、27、32 张的万用表误判，18 张照片的 23 条明确螺丝刀状态被判 unknown，以及真实遮挡子集的 unknown 召回 0/14。

## 1. 已定的方案（用户已确认）

不再采集数据，改为：**按摆放组 3 折交叉验证 + 用框合成成对训练图**。测试永远只用真实照片。

| 折 | 测试组 | 训练组 |
| --- | --- | --- |
| fold1 | l009（第 26–50 张） | 其余 |
| fold2 | l008（第 9–25 张） | 其余 |
| fold3 | l001–l007（第 1–8 张） | 其余 |

每个实验训练 3 次（每折一次），把三折测试集预测拼成覆盖全部 50 张的一份 `predictions.jsonl`，再用 `scripts/eval_predictions.py` 对 `real_v1.jsonl` 统一评测（bootstrap 按 scene_group）。

## 2. 任务清单（按顺序）

### 任务 A：B0 两项优化扫描（不需要训练，先做）

**完成。** 六个 L40 配置各有 100 条真实预测，结果在 `runs/comparison/comparison.md`。BF16 384²/512²/768² 的螺丝刀准确率分别为 63/100、67/100、91/100，误报齐全分别为 7/31、7/31、6/31。llama.cpp 在 512² 的 F16/Q8_0/Q4_K_M 决策准确率分别为 80/100、80/100、73/100，端到端延迟 p50 分别为 1.3821、1.2693、1.2110 秒。Q4 的 unknown 召回为 7/18，F16 为 2/18。结果来自 9 个摆放组，置信区间标为样本不足。

1. 分辨率：已有 512²；补跑 384²（`--max-pixels 147456`）和 768²（`589824`）：
   ```bash
   P=envs/qwen_vlm/bin/python
   $P scripts/predict_transformers.py --jsonl data/annotations/real_v1.jsonl --experiment-id B0_s001_r384 --max-pixels 147456
   $P scripts/eval_predictions.py --gold data/annotations/real_v1.jsonl --pred runs/B0_s001_r384/predictions.jsonl --out runs/B0_s001_r384/eval --experiment-id B0_s001_r384 --pixel-budget 384
   ```
2. 量化：用 GGUF（`artifacts/quantized/base/`）在 512² 下跑 F16 / Q8_0 / Q4_K_M：
   ```bash
   $P scripts/predict_llamacpp.py --jsonl data/annotations/real_v1.jsonl --experiment-id B0_s001_q4 \
      --model artifacts/quantized/base/base-q4_k_m.gguf --mmproj artifacts/quantized/base/mmproj-base-f16.gguf
   ```
   每个都接 `eval_predictions.py`（加 `--quantization Q4_K_M` 等）。最后 `scripts/eval_compare.py --runs-root runs --out runs/comparison`。
3. 重点观察：螺丝刀在不同分辨率下的准确率；Q4 相对 bf16 的退化（合成图上 Q4 只有 8/15 与 bf16 一致）。

### 任务 B：用框合成成对训练图

增强入口为 `scripts/data_make_augment.py`，实现位于 `src/qwen_vlm/data/augment.py`。从仓库根目录加载环境后运行以下命令。图片、样本、生成清单和抽查表保存在 Git 忽略的 `data/aug/v1/`。

```bash
source env/activate.sh
P=envs/qwen_vlm/bin/python
$P scripts/data_make_augment.py --seed 0
$P scripts/data_validate.py data/aug/v1/samples.jsonl
```

生成结果为 50 抹除对、50 遮挡对，共 200 张编辑图和 400 条 checklist。其中 33 对的真编辑抹掉万用表并保留独立线缆。每对使用同一套问法，新增困难负样本标签针对这 33 个真编辑。抽查表 `data/aug/v1/contact_sheet.jpg` 含 10 对共 20 张，曾发现第 25 张的万用表红色残影，修正后再次检查。遮罩纹理仍可辨识，需在报告中披露合成编辑与真实场景的差异。复现时参照 `data/aug/v1/manifest.json`。改种子或编辑参数时输出到另一个版本目录，以便已有 manifest 继续指向对应的图像。

- 输入：`real_v1.jsonl` + `annotations.json`。每张真实图生成若干编辑图，**继承原图的 `session_id` 和 `scene_group_id`**（保证交叉验证时合成图跟随原图所在的折，不泄漏）。
- 编辑类型：
  1. **抹除**：把某件 present 器材的框区域（适当外扩）用 `cv2.inpaint` 或白板纹理填掉 → 该类改为 `absent`。重点多做"抹掉万用表、保留线缆"（针对 B0 幻觉），打 `hard_negative_tags: ["cable_without_multimeter"]` 之类标签。
  2. **合成遮挡**：在某件器材上覆盖手形/纸张色块（覆盖大部分）→ 该类 `unknown`，`unknown_reason` 写"合成遮挡"。
  3. **诱饵编辑**（必须有，防止捷径）：在空白区域做同样的 inpaint 或色块，但不碰任何器材 → 状态不变。
- **成对**：每对 = 一张"诱饵编辑图" + 一张"真编辑图"（两张都有编辑痕迹，只差一个类别），`pair_group_id` 唯一。不要把原图放进新 pair（原图可能已在真实 pair 中，一张图只能属于一对）。
- 字段：`source: "synthetic_edit"`，`annotation_verified: false`，sample_id 形如 `s001_l008_v014e03_q01`（编号里不能有下划线）。图片缩到 1280×720 JPEG（q≈95）省磁盘。
- 问法/清单：复用 `configs/data/question_templates.yaml` 的 default 集合（参考 `src/qwen_vlm/data/convert.py` 的抽取方式）。
- 验收：`scripts/data_validate.py data/aug/v1/samples.jsonl` 0 错误；拼一张缩略图人工抽查 20 张编辑效果（抹除处不能残留明显器材，诱饵和真编辑痕迹看起来相似）；写 `tests/test_data_augment.py`（状态改变正确、诱饵不改状态、pair 恰差一类、scene_group 继承）。

### 任务 C：交叉验证划分与实验配置

1. `scripts/data_make_cv.py` 生成 `data/splits/cv3/fold{1,2,3}/`。固定划分已用真实数据和合成数据验证。三折各有 250、330、420 条混合训练样本和 50、34、16 条真实测试样本，覆盖真实金标准 100/100 条、50/50 张。每折 manifest 记录文件哈希及泄漏检查。
2. 实验（全部用同一 `max_steps`、有效 batch、LoRA 超参；**不在测试集上调参**，超参固定为 `configs/train/base.yaml`，只调 `max_steps` 使训练在 train 上收敛，三折一致）：

| ID | 训练数据 | mix 配置 |
| --- | --- | --- |
| B1 | 只用真实图 | `configs/train/B1.yaml`（无辅助、无加权） |
| M | 真实 + 合成 | `M.yaml`（pairs 辅助 + hard_negative_weight 3 + balance + unknown） |
| A1 | 真实 + 合成 | M 去掉成对辅助（`aux_presence: none`） |
| A2 | 真实 + 合成 | M 去掉困难负样本加权（`hard_negative_weight: 1.0`） |
| A3 | 真实 + 合成 | M 去掉 unknown 样本（`include_unknown: false`） |
| A4 | 只用真实图 | M 的 mix，但不加合成图（衡量合成数据本身的贡献） |

   B1 和 A4 从各折的 `train_real.jsonl` 读取 50、66、84 条真实训练样本。
3. 正式实验共用 **50 个优化步、有效 batch 16、种子 42**。两次 30 步试跑都只读取 fold1 训练集。B1 训练 loss 从 0.08168 降至约 0.0000062，且该折纯真实训练子集没有 unknown。混合 M 的三个 10 步窗口平均 loss 分别为 0.07808、0.02201、0.02199。M 三折重采样前的样本池分别为 360、468、604 条，50 步对应 800 次样本呈现，使最大折的预算比例接近 30 步试跑。试跑记录在 `work/cv3_pilot/{B1,M}_fold1/`。入口显式传 `--no-validation`，正式折外指标待测。
   ```bash
   source env/activate.sh
   P=envs/qwen_vlm/bin/python
   $P scripts/data_make_cv.py
   $P scripts/run_cv.py --max-steps 50
   $P scripts/eval_compare.py --runs-root runs --out runs/comparison
   ```
   `scripts/run_cv.py` 先核对三折固定分组、JSONL 和 manifest 哈希以及全部真实测试样本覆盖，再依次训练、预测、拼接和评测。折内记录在 `runs/cv3_folds/<ID>_fold<k>/`，汇总在 `runs/<ID>_cv3/`。可用原命令重跑并复用完成的 adapter 和预测。改变步数、配置或划分时使用新的 `--runs-root`。
4. 资源：项目盘已扩容，正式运行前约剩 59 GiB。每折 `save_total_limit` 为 1，折外预测核对后脚本自动清理 `checkpoints/`，保留 `adapter/`。

### 任务 D：汇总与文档

1. `scripts/eval_compare.py` 生成 B0/B1/M/A1–A4 对比表和优化矩阵表（`runs/comparison/`）。
2. 更新 `EXPERIMENTS.md`（加入 A4、交叉验证协议、合成数据说明）、`STATUS.md`、`reports/readiness.md`。
3. 报告必须写明的局限：只有 1 个批次、9 个摆放、50 张图；置信区间宽（标"样本不足"）；合成图只用于训练；交叉验证三折的测试组大小不均。
4. 测试状态：全量 `pytest -q tests` 111 个通过。服务器扫描的六份评测已归档。正式三折用同一命令恢复，并检查每个实验是否合并为覆盖 100 条金标准的折外预测。

### 任务 E：选定最终模型并部署（之后）

1. 将 `real_v1.jsonl` 和 `data/aug/v1/samples.jsonl` 合并为全量训练 JSONL，沿用正式实验的 M 配置和步数。用 `scripts/train.py --no-validation` 将 adapter 保存到 `runs/M_final/adapter/`。泛化指标使用前面的折外评测，最终模型用于部署演示。
2. `scripts/merge_lora.py` 合并，检查代表性输入的合并一致性，然后导出 F16/Q8_0/Q4_K_M 和 mmproj。项目盘已扩容到约 59 GiB 可用，合并模型（约 4.3 GiB）和 GGUF（约 6.8 GiB）直接放在 `artifacts/merged/final/` 和 `artifacts/quantized/final/`，具体命令见“本轮要做的事”第 4 步。
3. 用 `scripts/make_jetson_bundle.sh` 生成 `release/jetson_bundle_final/`，打包脚本对模型文件使用硬链接，要求 bundle 和 GGUF 在同一文件系统上。
4. 用户在 Jetson 上跑 `deploy/jetson/00`–`04`，最终 bundle 的 `02`–`04` 使用 `MODEL_TAG=final`。日志回传 `runs/jetson/`。默认 `03_bench.sh` 循环使用冒烟图片，可测延迟和资源。也可以对 `real_v1.jsonl` 在板端逐条推理，按部署说明保留 `sample_id`，回传预测与对应 `tegrastats`。对全量训练的最终模型，这批预测用于检查部署后的输出和性能。

## 3. 注意事项

- 每个 shell 先 `cd /root/gpufree-data/qwen_vlm && source env/activate.sh`，Python 一律用 `envs/qwen_vlm/bin/python`（不 source 会报 cuDNN 错误）。
- 不改 `contracts/`、`src/qwen_vlm/common/`、`src/qwen_vlm/inference/base.py`。
- 不编造结果；未跑的写"待测"。合成图/冒烟图的任务质量数字不进报告；板端冒烟图的延迟与资源数字仅在明确标注为计时用途时使用。
- 每完成一个任务跑 `pytest -q tests` 并 `git commit`；推送需用户输入 GitHub 令牌（见 `docs/git_workflow.md`）。服务器外网慢，代理常断，直连 GitHub 可用。
- 标注已全部完成，标注工具已关闭，不需要再启动。
