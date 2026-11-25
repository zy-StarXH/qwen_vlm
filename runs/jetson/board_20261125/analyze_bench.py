"""Summarize the Jetson board benchmark into summary.json / summary.md.

Reads the copied logs under this directory. Run with:
    PYTHONPATH=/home/jetson/qwen_vlm/src \
      /home/jetson/qwen_vlm/release/jetson_bundle_base/.venv/bin/python analyze_bench.py
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from qwen_vlm.evaluation.latency import latency_from_predictions
from qwen_vlm.evaluation.tegrastats import parse_tegrastats_log, summarize_tegrastats

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
BASELINE = ROOT / "baseline"
MODELS = Path("/home/jetson/qwen_vlm/release/jetson_bundle_base/models")
# File birth/mtime were reset when the logs were copied; read the originals for time windows.
ORIGINAL_BENCH = Path(
    "/home/jetson/qwen_vlm/release/jetson_bundle_base/logs/bench_20261125_165733"
)
TIME_DIR = ORIGINAL_BENCH if ORIGINAL_BENCH.exists() else BENCH


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_times(path: Path) -> tuple[datetime, datetime]:
    out = subprocess.check_output(["stat", "-c", "%W %Y", str(path)], text=True).split()
    birth, mtime = int(out[0]), int(out[1])
    if birth == 0:
        birth = mtime - 200
    return (
        datetime.fromtimestamp(birth - 2),
        datetime.fromtimestamp(mtime + 2),
    )


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


def config_summary(quant: str, pixels: str) -> dict:
    path = BENCH / f"{quant}_{pixels}.jsonl"
    records = read_jsonl(path)
    measured = [r for r in records if not r.get("warmup")]
    cold = [r for r in records if r.get("warmup") and (r["timings"].get("attempt0_prompt_tokens") or 0) > 100]
    errors = [r for r in records if r.get("error")]
    parsed = [r for r in measured if r.get("parsed")]
    tokens = sum(r.get("n_generated_tokens") or 0 for r in measured)
    generate_s = sum(r["timings"].get("attempt0_generate_s", 0.0) for r in measured)
    decisions = Counter((r.get("derived") or {}).get("decision") for r in measured)
    missing = Counter(
        item
        for r in measured
        for item in ((r.get("derived") or {}).get("missing_required") or [])
    )
    start, end = file_times(TIME_DIR / path.name)
    rows = parse_tegrastats_log((BENCH / "tegrastats.txt").read_text(encoding="utf-8"))
    tegra = summarize_tegrastats(rows, start_time=start.isoformat(sep=" "), end_time=end.isoformat(sep=" "))
    return {
        "quant": quant,
        "pixels": int(pixels),
        "image_max_tokens": {384: 144, 512: 256, 768: 576}[int(pixels)],
        "n_records": len(records),
        "n_measured": len(measured),
        "n_errors": len(errors),
        "n_parsed": len(parsed),
        "cold": {
            "n": len(cold),
            "total_s": stats([r["timings"]["total_s"] for r in cold]),
            "prefill_s": stats([r["timings"].get("attempt0_first_token_s", 0.0) for r in cold]),
        },
        "steady": {
            "total_s": stats([r["timings"]["total_s"] for r in measured]),
            "prefill_s": stats([r["timings"].get("attempt0_first_token_s", 0.0) for r in measured]),
            "generated_tokens": tokens,
            "generate_s": round(generate_s, 3),
            "tokens_per_s": round(tokens / generate_s, 2) if generate_s else None,
            "requests_per_s": round(len(measured) / sum(r["timings"]["total_s"] for r in measured), 4)
            if measured
            else None,
        },
        "decisions": dict(decisions),
        "missing_required_counts": dict(missing),
        "tegrastats": tegra,
    }


def main() -> None:
    rows = parse_tegrastats_log((BENCH / "tegrastats.txt").read_text(encoding="utf-8"))
    global_tegra = summarize_tegrastats(rows)
    configs = [config_summary(q, p) for q in ("q4_k_m", "q8_0") for p in ("384", "512", "768")]

    base = read_jsonl(BASELINE / "baseline_512_20261125_165719.jsonl")[0]
    baseline = {
        "sample_id": base["sample_id"],
        "total_s": round(base["timings"]["total_s"], 3),
        "prefill_s": round(base["timings"].get("attempt0_first_token_s", 0.0), 3),
        "generate_s": round(base["timings"].get("attempt0_generate_s", 0.0), 3),
        "generated_tokens": base.get("n_generated_tokens"),
        "parse_status": base.get("parse_status"),
        "derived": base.get("derived"),
        "raw_text": base.get("raw_text"),
    }

    f16 = {}
    for pixels in ("384", "512"):
        path = BENCH / f"f16_{pixels}.jsonl"
        records = read_jsonl(path)
        f16[pixels] = {
            "n_records": len(records),
            "n_errors": sum(1 for r in records if r.get("error")),
            "error_example": next((r.get("error") for r in records if r.get("error")), None),
        }
    oom_lines = (BENCH / "f16_oom_kernel.log").read_text(encoding="utf-8").strip().splitlines()

    demo = json.loads((ROOT / "demo/image_mode_check.json").read_text(encoding="utf-8"))

    model_files = {}
    for name in ("base-f16.gguf", "base-q8_0.gguf", "base-q4_k_m.gguf", "mmproj-base-f16.gguf"):
        path = MODELS / name
        model_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": subprocess.check_output(["sha256sum", str(path)], text=True).split()[0],
        }

    summary = {
        "device": "Jetson Orin Nano 8GB (Engineering Reference Developer Kit Super)",
        "l4t": "R36.4.3",
        "cuda": "12.6.11",
        "power_mode": "MAXN_SUPER",
        "llamacpp": "b12425 / 18b74fa41dcac1f55ae1371fe58153f7225941f6",
        "models": "Qwen3-VL-2B-Instruct base (official F16 GGUF from ModelScope; quantized on board)",
        "model_files": model_files,
        "baseline_02": baseline,
        "bench": configs,
        "f16": {"status": "oom_killed", "configs": f16, "kernel": oom_lines},
        "tegrastats_global": global_tegra,
        "demo": demo,
        "notes": [
            "cold = first presentation of each of the three smoke images (prompt_tokens > 100); n=3.",
            "steady = 100 measured requests; repeated identical images hit llama-server prompt cache, so prefill is ~0.03 s.",
            "per-config tegrastats windows start when the JSONL is created (after server health) and exclude model load.",
            "real-photo accuracy was not measured: data/raw/session_001 is absent from this clone and the final adapter is not included.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fnum(value, digits=1):
        return "未测" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Jetson Orin Nano 8GB 板端实测（2026-11-25）",
        "",
        "- 设备：Jetson Orin Nano 8GB Engineering Reference Developer Kit Super，L4T R36.4.3，CUDA 12.6.11，MAXN_SUPER",
        "- llama.cpp：b12425 / 18b74fa4，CUDA 编译（sm_87）",
        "- 模型：Qwen3-VL-2B-Instruct 原始权重，官方 F16 GGUF（ModelScope，SHA256 27b80f2e…2595），板上用 b12425 的 llama-quantize 转 Q8_0/Q4_K_M",
        "- 说明：最终微调 adapter 与真实照片不在本机克隆中；本表是原始模型（B0 权重）的板端部署与性能实测",
        "",
        "## 单张图片冷启动（02 baseline，Q4_K_M，512²）",
        "",
        f"- 端到端 {baseline['total_s']} s，prefill {baseline['prefill_s']} s，generate {baseline['generate_s']} s，生成 {baseline['generated_tokens']} token",
        f"- 解析：{baseline['parse_status']}；derived：{baseline['derived']}",
        "",
        "## 量化 × 像素档",
        "",
        "| 量化 | 像素 | 视觉 token 上限 | 冷启动 p50 (n=3) | 冷 prefill p50 | 稳态 p50 / p95 (n=100) | 稳态 token/s | 解析 | 决策分布 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in configs:
        cold = item["cold"]["total_s"]
        cold_first = item["cold"]["prefill_s"]
        steady = item["steady"]["total_s"]
        lines.append(
            f"| {item['quant']} | {item['pixels']}² | {item['image_max_tokens']} | "
            f"{cold['p50']} s | {cold_first['p50']} s | {steady['p50']} / {steady['p95']} s | "
            f"{item['steady']['tokens_per_s']} | {item['n_parsed']}/{item['n_measured']} | "
            f"{item['decisions']} |"
        )
    lines += [
        "",
        "冷启动是每张冒烟图第一次出现时的 3 次请求；稳态是 100 次重复请求，会命中 llama-server 的 prompt cache，"
        "所以稳态 total_s 主要是生成耗时，不能当作全新照片的端到端延迟。",
        "",
        "## 每档内存与功耗（每档 JSONL 写入窗口，不含模型加载）",
        "",
        "| 量化 | 像素 | RAM 均值 / 峰值 (MB) | SWAP 峰值 (MB) | VDD_IN 均值 / 峰值 (mW) | GPU 频率均值 | 最高温度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in configs:
        t = item["tegrastats"]
        power = t["power_mw"].get("VDD_IN", {}).get("instant_mw", {})
        peak_temp = max((v["peak"] for v in t["temperatures_c"].values() if v["peak"]), default=None)
        lines.append(
            f"| {item['quant']} | {item['pixels']}² | "
            f"{fnum(t['ram_used_mb']['mean'], 0)} / {fnum(t['ram_used_mb']['peak'], 0)} | "
            f"{fnum(t['swap_used_mb']['peak'], 0)} | "
            f"{fnum(power.get('mean'), 0)} / {fnum(power.get('peak'), 0)} | "
            f"{fnum(t['gr3d_freq_pct']['mean'])}% | {fnum(peak_temp)} °C |"
        )
    lines += [
        "",
        "## F16 语言权重",
        "",
        "- F16（3.21 GiB）+ F16 mmproj（0.76 GiB）+ KV/compute 在 8GB 统一内存上被 OOM killer 终止。",
        f"- 内核记录：`{oom_lines[-1] if oom_lines else '未取到'}`",
        "- Q8_0（1.71 GiB）与 Q4_K_M（1.03 GiB）在同样条件下均完成 3 档像素的 100 次请求。",
        "",
        "## 全流程内存与功耗（tegrastats，整段 9 档）",
        "",
        f"- RAM 已用：均值 {fnum(global_tegra['ram_used_mb']['mean'], 0)} MB，峰值 {fnum(global_tegra['ram_used_mb']['peak'], 0)} MB / {fnum(global_tegra['ram_total_mb']['peak'], 0)} MB",
        f"- SWAP 已用峰值：{fnum(global_tegra['swap_used_mb']['peak'], 0)} MB",
        f"- VDD_IN：均值 {fnum(global_tegra['power_mw'].get('VDD_IN', {}).get('instant_mw', {}).get('mean'), 0)} mW，峰值 {fnum(global_tegra['power_mw'].get('VDD_IN', {}).get('instant_mw', {}).get('peak'), 0)} mW",
        f"- GPU 频率占比均值 {fnum(global_tegra['gr3d_freq_pct']['mean'])}%，EMC 均值 {fnum(global_tegra['emc_freq_pct']['mean'])}%",
        "",
        "## 演示界面（04，Q4_K_M，512²）",
        "",
        "| 图片 | 墙钟 (s) | 推理 (s) | 决策 | 缺失/无法确认 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in demo["rows"]:
        status = row["status"]
        infer = status.split("总耗时 ")[1].split(" 秒")[0] if "总耗时 " in status else "?"
        decision = "unknown" if "无法确认" in row["missing"] else ("incomplete" if row["missing"] and not row["missing"].startswith("（") else "complete")
        lines.append(f"| {Path(row['image']).name} | {row['wall_s']} | {infer} | {decision} | {row['missing']} |")
    lines += [
        "",
        "本机没有 `/dev/video*`，摄像头链路未测；演示使用「读取保存图片」模式，调用 Gradio `/infer` 接口完成同一推理路径。",
        "",
        "## 未测项",
        "",
        "- 板端真实照片准确率：`data/raw/session_001/` 与最终 adapter 都不在本机克隆中。",
        "- 摄像头 FPS、现场演示视频：无摄像头设备。",
        "- F16 的精度：语言权重在 8GB 上 OOM，未进入推理。",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ROOT / "summary.json", ROOT / "summary.md")


if __name__ == "__main__":
    main()
