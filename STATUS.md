# 项目状态

更新于 2026-11-26。服务器第 1–5 步完成。最终 Q4_K_M bundle 与 LoRA adapter 已从云端恢复到 Jetson，逐字节对照 `runs/M_final/bundle_SHA256SUMS` 通过；板端完成最终 M 的 Q4 三档计时、50 张真实照片部署检查和四张演示图。原始权重 Q4/Q8 三档扫描与 F16 OOM 记录保留在 `runs/jetson/board_20261125/`；最终 M 证据见 `runs/jetson/final_20261126/`。最终 Q8_0/F16 未上传，摄像头与录像仍缺。

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
| P4 | 量化与像素档的板端实测 | 最终 M Q4 已验证；Q8/F16 未上传，F16 原始权重 OOM | 最终 Q4 三档各 100 次请求全部解析，稳态 p50 1.21–1.29 s；真实照片 100 条 p50 3.01 s。见 `runs/jetson/final_20261126/summary.md`。原始权重 Q4/Q8 参考见 `runs/jetson/board_20261125/` |
| P5 | 最终 Q4_K_M 图片模式 | 服务器与板端均已验证 | 服务器四类样例见 `runs/M_final/demo/image_mode_check.json`；板端 v001/v014/v040/v041 经 Gradio `/infer` 决策与服务器一致，见 `runs/jetson/final_20261126/demo/image_mode_check.json` |
| P5 | 摄像头、现场演示与视频录制 | 无摄像头设备，待测、待录 | `/dev/video*` 不存在；`deploy/jetson/04_app.sh` 的图片模式已跑通 |
| P6 | 英文报告、中文逐段校对版、演示脚本、答辩提纲 | 完成 | `reports/technical_report.md`、`reports/technical_report.zh.md`、`reports/demo_script.md`、`reports/defense_outline.md` |
