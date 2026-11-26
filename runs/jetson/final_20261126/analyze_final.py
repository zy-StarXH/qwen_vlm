"""Summarize the final-M Jetson board run (2026-11-26).

Run with:
  PYTHONPATH=/home/jetson/qwen_vlm/src python3 analyze_final.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from qwen_vlm.common.derive import derive
from qwen_vlm.evaluation.latency import latency_from_predictions, throughput_from_predictions
from qwen_vlm.evaluation.tegrastats import parse_tegrastats_log, summarize_tegrastats

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
REAL = ROOT / "real_eval"
BUNDLE_BENCH = Path("/home/jetson/qwen_vlm/release/jetson_bundle_base/logs/bench_20261126_193326")
MODELS = Path("/home/jetson/qwen_vlm/release/jetson_bundle_final/models")
GOLD = Path("/home/jetson/qwen_vlm/data/annotations/real_v1.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": round(sum(ordered) / n, 3),
        "p50": round(ordered[n // 2], 3),
        "p95": round(ordered[min(n - 1, int(round(0.95 * (n - 1))))], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def file_times(path: Path) -> tuple[datetime, datetime]:
    out = subprocess.check_output(["stat", "-c", "%W %Y", str(path)], text=True).split()
    birth, mtime = int(out[0]), int(out[1])
    if birth == 0:
        birth = mtime - 200
    return datetime.fromtimestamp(birth - 2), datetime.fromtimestamp(mtime + 2)


def bench_config(pixels: str) -> dict:
    path = BENCH / f"q4_k_m_{pixels}.jsonl"
    records = read_jsonl(path)
    measured = [r for r in records if not r.get("warmup")]
    cold = [r for r in records if r.get("warmup") and (r["timings"].get("attempt0_prompt_tokens") or 0) > 100]
    tokens = sum(r.get("n_generated_tokens") or 0 for r in measured)
    generate_s = sum(r["timings"].get("attempt0_generate_s", 0.0) for r in measured)
    start, end = file_times(BUNDLE_BENCH / path.name)
    rows = parse_tegrastats_log((BENCH / "tegrastats.txt").read_text(encoding="utf-8"))
    tegra = summarize_tegrastats(rows, start_time=start.isoformat(sep=" "), end_time=end.isoformat(sep=" "))
    return {
        "pixels": int(pixels),
        "image_max_tokens": {384: 144, 512: 256, 768: 576}[int(pixels)],
        "n_measured": len(measured),
        "n_parsed": sum(1 for r in measured if r.get("parsed")),
        "n_errors": sum(1 for r in records if r.get("error")),
        "cold_total_s": stats([r["timings"]["total_s"] for r in cold]),
        "cold_prefill_s": stats([r["timings"].get("attempt0_first_token_s", 0.0) for r in cold]),
        "steady_total_s": stats([r["timings"]["total_s"] for r in measured]),
        "steady_prefill_s": stats([r["timings"].get("attempt0_first_token_s", 0.0) for r in measured]),
        "tokens_per_s": round(tokens / generate_s, 2) if generate_s else None,
        "requests_per_s": round(len(measured) / sum(r["timings"]["total_s"] for r in measured), 4),
        "tegrastats": tegra,
    }


def metrics(path: Path) -> dict:
    c = json.loads(path.read_text(encoding="utf-8"))["checklist"]
    return {
        "decision_accuracy": c["decision_accuracy"],
        "false_complete": c["false_complete_rate"],
        "unknown_recall": c["unknown_recall"],
        "state_accuracy": c["state_accuracy"]["overall"],
        "macro_f1": c["macro_f1"],
        "json_valid": c["json_valid_rate"],
    }


def fmt_rate(entry: dict) -> str:
    return f"{entry['numerator']}/{entry['denominator']}"


def main() -> None:
    configs = [bench_config(p) for p in ("384", "512", "768")]

    board_metrics = metrics(REAL / "board_real_eval/metrics.json")
    server_metrics = metrics(REAL / "server_l40_reference_eval/metrics.json")
    board_pred = {r["sample_id"]: r for r in read_jsonl(REAL / "predictions_real.jsonl")}
    server_pred = {r["sample_id"]: r for r in read_jsonl(Path("/home/jetson/qwen_vlm/runs/M_final/q4_k_m/predictions.jsonl"))}
    gold = {r["sample_id"]: r for r in read_jsonl(GOLD)}
    state_agree = sum(1 for k in board_pred if board_pred[k].get("parsed") == server_pred[k].get("parsed"))
    raw_agree = sum(1 for k in board_pred if board_pred[k].get("raw_text") == server_pred[k].get("raw_text"))
    decision_agree = 0
    for k in board_pred:
        if derive(board_pred[k]["parsed"], gold[k]["required_items"])["decision"] == derive(
            server_pred[k]["parsed"], gold[k]["required_items"]
        )["decision"]:
            decision_agree += 1

    real = latency_from_predictions(read_jsonl(REAL / "predictions_real.jsonl"))
    throughput = throughput_from_predictions(read_jsonl(REAL / "predictions_real.jsonl"))
    resources = json.loads((REAL / "board_real_eval/metrics.json").read_text(encoding="utf-8"))["resources"]
    vdd = resources["power_mw"].get("VDD_IN", {}).get("instant_mw", {})

    board_demo = json.loads((ROOT / "demo/image_mode_check.json").read_text(encoding="utf-8"))["rows"]
    server_demo = json.loads(
        Path("/home/jetson/qwen_vlm/runs/M_final/demo/image_mode_check.json").read_text(encoding="utf-8")
    )["cases"]
    server_by_photo = {c["photo"]: c for c in server_demo}
    req = ["screwdriver", "pliers", "cable_bundle", "multimeter", "hammer"]
    demo_rows = []
    for row in board_demo:
        text = row["answer_raw"].split("原始输出：", 1)[1].strip()
        if text.startswith("```"):
            text = text[3:]
            if text.startswith("json"):
                text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        states = json.loads(text.strip())["item_states"]
        decision = derive(states, req)["decision"]
        infer_s = re.search(r"总耗时 ([0-9.]+) 秒", row["status"])
        server = server_by_photo.get(row["id"], {})
        demo_rows.append(
            {
                "photo": row["id"],
                "board_decision": decision,
                "server_decision": server.get("ui_decision"),
                "board_infer_s": float(infer_s.group(1)) if infer_s else None,
                "server_infer_s": server.get("ui_total_s"),
                "board_missing": row["missing"],
            }
        )

    model_files = {}
    for name in ("final-q4_k_m.gguf", "mmproj-final-f16.gguf", "adapter/adapter_model.safetensors"):
        path = MODELS / name
        model_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": subprocess.check_output(["sha256sum", str(path)], text=True).split()[0],
        }

    summary = {
        "date": "2026-11-26",
        "device": "Jetson Orin Nano 8GB Engineering Reference Developer Kit Super",
        "l4t": "R36.4.3",
        "cuda": "12.6.11",
        "power_mode": "MAXN_SUPER",
        "llamacpp": "b12425 / 18b74fa41dcac1f55ae1371fe58153f7225941f6",
        "model": "M final (LoRA merged), Q4_K_M language + F16 mmproj",
        "model_files": model_files,
        "bench": configs,
        "real_photo_deployment_check": {
            "purpose": "deployment/export check on the 50 training photographs; NOT a generalization metric",
            "n_checklist": 100,
            "board": board_metrics,
            "server_l40": server_metrics,
            "board_vs_server_parsed_state_agreement": f"{state_agree}/{len(board_pred)}",
            "board_vs_server_raw_text_agreement": f"{raw_agree}/{len(board_pred)}",
            "board_vs_server_decision_agreement": f"{decision_agree}/{len(board_pred)}",
            "latency": real,
            "throughput": throughput,
            "resources": resources,
        },
        "demo": demo_rows,
        "notes": [
            "Bench inputs are the three synthetic smoke images; only timing/resources are used.",
            "Real-photo metrics use photographs that were part of final training; they verify deployment, not generalization.",
            "The held-out three-fold result (86/100, 2/31, 1/18) remains the generalization evidence.",
            "Final Q8_0/F16 GGUFs were not uploaded; only the verified Q4_K_M bundle is available.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fnum(v, d=2):
        return "未测" if v is None else f"{v:.{d}f}"

    lines = [
        "# Jetson Orin Nano 8GB 最终 M 板端实测（2026-11-26）",
        "",
        "- 设备：Jetson Orin Nano 8GB Engineering Reference Developer Kit Super，L4T R36.4.3，CUDA 12.6.11，`MAXN_SUPER`",
        "- llama.cpp：b12425 / `18b74fa4`，板上 CUDA sm_87 构建",
        f"- 模型：最终 M 的 Q4_K_M 语言 GGUF（SHA256 `{model_files['final-q4_k_m.gguf']['sha256'][:12]}…`）+ F16 mmproj，均由服务器最终 bundle 上传并逐字节校验",
        "- 真实照片：`data/raw/session_001/` 50 张，覆盖 `real_v1.jsonl` 的 50 个 image_path",
        "",
        "## 1. 最终 M 的 3 档像素计时（冒烟图，仅用于性能）",
        "",
        "| 像素 | 视觉 token 上限 | 冷启动 p50 / p95 (n=3) | 冷 prefill p50 | 稳态 p50 / p95 (n=100) | 生成 token/s | 峰值 RAM | 峰值 VDD_IN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in configs:
        t = c["tegrastats"]
        vdd_c = t["power_mw"].get("VDD_IN", {}).get("instant_mw", {})
        lines.append(
            f"| {c['pixels']}² | {c['image_max_tokens']} | "
            f"{c['cold_total_s']['p50']:.2f} / {c['cold_total_s']['p95']:.2f} s | "
            f"{c['cold_prefill_s']['p50']:.2f} s | "
            f"{c['steady_total_s']['p50']:.2f} / {c['steady_total_s']['p95']:.2f} s | "
            f"{c['tokens_per_s']:.1f} | {fnum(t['ram_used_mb']['peak'], 0)} MB | {fnum(vdd_c.get('peak'), 0)} mW |"
        )
    lines += [
        "",
        "每档 5 次预热 + 100 次正式请求（共 315 次）全部解析；其中冷启动是每档 3 张冒烟图第一次出现，共 9 次。稳态请求重复三张冒烟图，会命中 llama-server 的 prompt cache；真实照片评测不受这一点影响。",
        "",
        "## 2. 真实照片部署检查（最终 M，Q4_K_M，512²）",
        "",
        "这 50 张照片全部参与过最终训练，因此本节是**部署/导出检查**，不是泛化指标。泛化证据仍是 L40 的折外三折结果（86/100、2/31、1/18）。",
        "",
        "| 指标 | Jetson 板端 | L40 服务器参考 |",
        "| --- | ---: | ---: |",
        f"| 决策准确率 | {fmt_rate(board_metrics['decision_accuracy'])} | {fmt_rate(server_metrics['decision_accuracy'])} |",
        f"| 误报齐全率 | {fmt_rate(board_metrics['false_complete'])} | {fmt_rate(server_metrics['false_complete'])} |",
        f"| unknown 召回 | {fmt_rate(board_metrics['unknown_recall'])} | {fmt_rate(server_metrics['unknown_recall'])} |",
        f"| 三状态准确率 | {fmt_rate(board_metrics['state_accuracy'])} | {fmt_rate(server_metrics['state_accuracy'])} |",
        f"| 宏观 F1 | {board_metrics['macro_f1']['value']:.3f} | {server_metrics['macro_f1']['value']:.3f} |",
        f"| JSON 有效 | {fmt_rate(board_metrics['json_valid'])} | {fmt_rate(server_metrics['json_valid'])} |",
        "",
        f"- 逐样本一致性：parsed 状态 {state_agree}/{len(board_pred)}，raw 文本 {raw_agree}/{len(board_pred)}，决策 {decision_agree}/{len(board_pred)}。",
        f"- 端到端 p50 / p95：{real['total_s']['p50']:.2f} / {real['total_s']['p95']:.2f} s；prefill p50 {real['first_token_s']['p50']:.2f} s；生成 p50 {real['generate_s']['p50']:.2f} s。",
        f"- 完整请求 {throughput['checklist_requests_per_s']['value']:.3f} request/s；生成 {throughput['generated_tokens_per_s']['value']:.1f} token/s。",
        f"- 资源：RAM 均值/峰值 {resources['ram_used_mb']['mean']:.0f}/{resources['ram_used_mb']['peak']:.0f} MB（共 {resources['ram_total_mb']['peak']:.0f} MB）；VDD_IN 均值/峰值 {vdd.get('mean', 0):.0f}/{vdd.get('peak', 0):.0f} mW；GPU 频率均值 {resources['gr3d_freq_pct']['mean']:.1f}%。",
        "- 这 100 条请求使用 1920×1080 的真实照片，包含图像加载与编码；延迟明显高于三张冒烟图的稳态值。资源窗口包含桌面与 agent 会话，swap 已用量也包含之前 benchmark 留下的状态。",
        "",
        "## 3. 演示界面（最终 M，Q4_K_M，512²，四张真实照片）",
        "",
        "| 照片 | 板端决策 | L40 决策 | 板端推理 (s) | L40 推理 (s) | 板端缺失/无法确认 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for d in demo_rows:
        lines.append(
            f"| {d['photo']} | {d['board_decision']} | {d['server_decision']} | {d['board_infer_s']} | {d['server_infer_s']} | {d['board_missing']} |"
        )
    lines += [
        "",
        "四个样例的决策与服务器一致；板端推理约 3.2–3.6 s，L40 约 1.2 s。",
        "",
        "## 4. 未完成/未上传",
        "",
        "- 最终 Q8_0 与 F16 GGUF 没有上传；本轮只验证了 Q4_K_M。F16 在 8 GB 上的 OOM 结论来自 2026-11-25 的原始权重测试，最终 F16 未测。",
        "- 没有 `/dev/video*`，摄像头 FPS、现场演示和视频仍未完成。",
        "- 最终 M 的板端真实照片指标是训练集部署检查，不能替代折外泛化结果。",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ROOT / "summary.md", ROOT / "summary.json")


if __name__ == "__main__":
    main()
