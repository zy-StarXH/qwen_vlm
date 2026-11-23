# 基于 Qwen3-VL-2B 的实验室器材完整性点检（Jetson Orin Nano 8GB）

固定俯视摄像头拍实验台，用户给出所需器材清单（螺丝刀、钳子、线缆、万用表、锤子中的任意子集），系统判断是否齐全、缺什么；画面不足以判断时回答"无法确认"。模型输出五类三状态 JSON，完整性结论由 `src/qwen_vlm/common/derive.py` 确定性推导。

- **接手请先读 `HANDOFF.md` 和 `docs/NEXT_STEPS.md`**
- 课程要求和证据对应：`docs/coursework_requirements.md`
- 当前进度：`STATUS.md`；就绪情况：`reports/readiness.md`；实验登记：`EXPERIMENTS.md`
- 模型原理：`docs/model_principles.md`；冻结接口：`contracts/`

## 目录

```text
env/          环境说明、activate.sh、lock 文件       envs/       conda 环境（不入库）
contracts/    类别表、schema、指标定义、接口约定    configs/    数据/训练/推理/量化配置
src/qwen_vlm/ common 推理 data training evaluation app
scripts/      全部命令行入口                        tests/      pytest
data/         smoke/（合成冒烟）raw/（真实照片）splits/（划分）annotations/
artifacts/    base/（基础模型）adapters/ merged/ quantized/（GGUF）
deploy/jetson/ 板端脚本与说明                        runs/       每次运行的日志、预测、评测
reports/      环境、就绪、结果与报告                 release/    交付包（Jetson bundle 等）
```

## 0. 环境

每个 shell 先执行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
```

`activate.sh` 会把所有缓存和临时文件指向项目目录，修复系统 cuDNN 冲突，设置 pip 镜像。用 `envs/qwen_vlm/bin/python` 运行一切 Python 脚本。环境的构建过程和网络情况见 `env/environment.yml` 与 `reports/environment_server.md`。

```bash
envs/qwen_vlm/bin/python -m pytest -q tests          # 全部单元测试
envs/qwen_vlm/bin/python scripts/download_base_model.py --verify-only   # 校验基础模型
envs/qwen_vlm/bin/python scripts/smoke_b0.py         # 基础模型冒烟（合成图）
```

## 1. 数据

采集与标注按 `docs/data_collection.md`、`docs/annotation_guide.md`。照片放到 `data/raw/<session>/<layout>/<variant>.jpg`，标注表模板在 `data/annotations/templates/annotation_template.csv`（编号不要含下划线）。

也可以用网页标注工具逐张标注（浏览器打开 `localhost:7861`），结果写入 `data/annotations/<批次>.csv`：

```bash
P=envs/qwen_vlm/bin/python
$P scripts/data_label_tool.py data/raw/session_001                 # 生成 data/annotations/s001.csv
$P scripts/data_csv_to_jsonl.py --csv data/annotations/s001.csv --out data/annotations/real_v1.jsonl --source real_lab --seed 0
$P scripts/data_validate.py data/annotations/real_v1.jsonl
```

当前 `session_001` 已完成标注。使用器材框生成成对训练图，再创建固定三折：

```bash
$P scripts/data_make_augment.py --seed 0
$P scripts/data_validate.py data/aug/v1/samples.jsonl
$P scripts/data_make_cv.py
```

合成冒烟数据（只用于打通流程）：`$P scripts/data_make_synthetic.py`，产物在 `data/smoke/synthetic/`。

## 2. 基线 B0 与训练

```bash
# 按 docs/NEXT_STEPS.md 记录的统一训练预算运行 B1/M/A1–A4 的固定三折。
$P scripts/run_cv.py --max-steps 50

# 汇总对比表
$P scripts/eval_compare.py --runs-root runs --out runs/comparison
```

B0 在 50 张真实照片上的 512² 基线已归档于 `runs/B0_s001/`，其余分辨率与量化档见 `runs/comparison/`。每折 adapter 位于 `runs/cv3_folds/<ID>_fold<k>/adapter/`，每个实验的三折汇总结果位于 `runs/<ID>_cv3/eval/`。

冒烟训练：`$P scripts/train.py --config configs/train/smoke.yaml`（合成数据，约 30 步，显存峰值约 5.5GiB）。

## 3. 合并与转换 GGUF

```bash
# 将 STAGING 设为有约 16 GiB 可用空间的临时目录。目录规划见 docs/NEXT_STEPS.md。
# 用任务 E 的全量训练 adapter 检查合并前后贪心输出。
$P scripts/merge_lora.py --adapter runs/M_final/adapter --sample-jsonl data/annotations/real_v1.jsonl \
   --merged-dir "$STAGING/merged" --merge-check runs/M_final/merge_check.json

# 服务器上构建 llama.cpp（只需一次），再转换：F16 语言 GGUF + F16 mmproj + Q8_0 + Q4_K_M
bash scripts/export_build_llamacpp_server.sh
bash scripts/export_gguf.sh "$STAGING/merged" "$STAGING/gguf" final

# 服务器上的部署冒烟：像素档对照、与 Transformers 的输出对照、演示界面
$P scripts/deploy_smoke.py

# 用 GGUF 对任意 JSONL 批量推理（自动启动/关闭本机 llama-server；也可用 --server 连接已运行的服务），输出格式同 predict_transformers.py
$P scripts/predict_llamacpp.py --jsonl data/annotations/real_v1.jsonl --experiment-id M_final_q4_512 \
   --model "$STAGING/gguf/final-q4_k_m.gguf" --mmproj "$STAGING/gguf/mmproj-final-f16.gguf" \
   --max-pixels 262144 --output runs/deploy_final/q4_512/predictions.jsonl
```

llama.cpp 固定为 tag `b12425`（commit `18b74fa4`），依据见 `deploy/jetson/LLAMACPP_VERSION.md`。

## 4. Jetson 部署

```bash
# 打包（默认只带 Q4_K_M；做量化对比时带全三档）
INCLUDE_QUANTS=q4_k_m,q8_0,f16 bash scripts/make_jetson_bundle.sh "$STAGING/gguf" "$STAGING/jetson_bundle_final"
```

把 bundle 拷到板子上，按 `deploy/jetson/README.md` 依次运行 `00_probe.sh` 到 `04_app.sh`，再把 `logs/` 拷回服务器的 `runs/jetson/`。板端日志评测：

```bash
$P scripts/eval_predictions.py --gold <对应 JSONL> --pred runs/jetson/<bench>/<config>.jsonl \
   --out runs/jetson/<config>/eval --quantization Q4_K_M --pixel-budget 512 --warmup 5 \
   --tegrastats runs/jetson/<bench>/tegrastats.log
```

演示界面（服务器或板端均可，需先启动 llama-server）：

```bash
$P -m qwen_vlm.app.app --server http://127.0.0.1:8080 --pixels 512 --camera 0
```

实时摄像头 + 空格触发推理的独立窗口程序（同一 llama-server；本机也可用 `--backend transformers` 直接加载模型）：

```bash
$P scripts/live_check.py --backend llamacpp --server http://127.0.0.1:8080 --pixels 512
```

用法与排错见 `docs/live_check.md`。

## 版本固定

- 基础模型：`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，逐文件 SHA256 见 `artifacts/base/manifest.json`
- llama.cpp：`b12425` / `18b74fa41dcac1f55ae1371fe58153f7225941f6`
- Python 环境：`env/environment.lock.yml`、`env/pip_freeze_qwen_vlm.txt`
- 每次运行：`runs/<experiment_id>/run_manifest.json`（git commit、数据哈希、配置、种子、步数）
