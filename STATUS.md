# 项目状态

更新于 2026-11-25。服务器第 1–5 步完成。Jetson Orin Nano 8GB 已用本机重建的原始权重 bundle 完成 00–04：llama.cpp b12425 构建通过，Q4_K_M/Q8_0 各完成 3 档像素扫描（共 6 档），F16 在 8 GB 上被 OOM killer 终止；04 的已保存图片模式已验证。最终微调 adapter、真实照片板端评测、摄像头与录像仍缺。板端证据见 `runs/jetson/board_20261125/`，复现见 [本轮计划](docs/NEXT_STEPS.md)。

| 阶段 | 内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| P0 | 课程要求核对、环境、接口冻结 | 已验证 | `docs/coursework_requirements.md`，`reports/environment_server.md`，`contracts/` |
| P1 | 基础模型单图推理与结构化输出（服务器） | 已验证（合成图） | `runs/smoke_b0/` |
| P1 | 云端转换 GGUF，与 Transformers 对照 | 已验证（合成图） | `artifacts/quantized/base/manifest.json`，`runs/deploy_smoke/` |
| P1 | 板端试跑原模型 | 已验证（原始权重，本机重建 bundle） | 00–02 PASS；Q4_K_M 512² 冷启动端到端 2.68 s。见 `runs/jetson/board_20261125/` |
| P2 | 真实照片采集、状态及器材框标注 | 已验证 | 50 张、100 条 checklist，`data/annotations/` |
| P2 | 器材框增强 | 已生成 | 100 对、200 张编辑图，`data/aug/v1/manifest.json` |
| P2 | 固定三折划分及泄漏检查 | 已验证 | `data/splits/cv3/`，100 条真实测试样本恰好覆盖一次 |
| P3 | B0 真实评测 | 已验证 | `runs/B0_s001/eval/` |
| P3 | B0 服务器分辨率和量化扫描 | 已验证 | L40 上 BF16 三档、GGUF 三档，`runs/comparison/` |
| P3 | LoRA 训练与固定三折入口，B1/M/A1–A4 配置 | B1、M 的真实训练子集试跑已通过 | `scripts/run_cv.py`，`work/cv3_pilot/` |
| P3 | 正式训练与消融 | 18/18 折已复核 | 每折 50 步，B1、M、A1–A4 各 100 条真实折外预测。详见 `runs/cv3_folds/`、`runs/*_cv3/eval/` |
| P3 | 对比与部署选择 | 完成 | B0 误报齐全 7/31，六种 LoRA 方法均为 2/31。M 决策准确率 86/100，unknown 召回 1/18，按规则保留 M。见 `EXPERIMENTS.md` |
| P3 | 鲁棒性条件 | 27/27 组已复核 | B0、B1、M 各 9 档，每档 100 条。汇总见 `EXPERIMENTS.md`，产物见 `runs/robust/` |
| P3 | 评测代码（全部任务指标、bootstrap、出表） | 已验证（手算单测） | `src/qwen_vlm/evaluation/`，`tests/test_metrics.py` |
| P4 | 最终 M、合并与导出 | 已验证 | 全量 500 条训练 50 步，合并一致，Q4/Q8 各 100 条导出检查，部署包 91 项 SHA256 校验通过。见 `runs/M_final/` |
| P4 | 量化与像素档的板端实测 | Q4/Q8 原始权重已验证；F16 OOM；最终 M 板端待测 | 6 个配置各 100 次请求全部解析。Q4 稳态 p50 1.62–1.71 s；F16 语言权重被 OOM killer 终止。见 `runs/jetson/board_20261125/summary.md` |
| P5 | 最终 Q4_K_M 图片模式 | 服务器已验证；板端原始权重已验证 | 服务器四类样例见 `runs/M_final/demo/image_mode_check.json`；板端三张冒烟图经 Gradio `/infer` 全部解析，见 `runs/jetson/board_20261125/demo/image_mode_check.json` |
| P5 | 摄像头、现场演示与视频录制 | 无摄像头设备，待测、待录 | `/dev/video*` 不存在；`deploy/jetson/04_app.sh` 的图片模式已跑通 |
| P6 | 英文报告、中文逐段校对版、演示脚本、答辩提纲 | 完成 | `reports/technical_report.md`、`reports/technical_report.zh.md`、`reports/demo_script.md`、`reports/defense_outline.md` |
