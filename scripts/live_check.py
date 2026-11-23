#!/usr/bin/env python3
"""实时摄像头 + 键盘触发一次 Qwen3-VL 器材完整性点检。

启动后显示摄像头实时画面；按空格（或 Enter）把当前帧送去模型推理一次，
结论（齐全 / 不齐全 / 无法确认）直接画在画面上，同时打印到终端。
默认按五类器材全部在清单内判断，可用数字键 1-5 临时增减清单。

两种后端：
  * llamacpp    连接已经启动的 llama-server（Jetson 部署用，推荐）
  * transformers 本机直接加载 Qwen3-VL-2B（开发机用，可加 LoRA adapter）
  * none        只测试相机和界面，不加载模型

例：
  # Jetson：先按 deploy/jetson/04_app.sh 的方式启动 llama-server，再运行
  python3 scripts/live_check.py --backend llamacpp --server http://127.0.0.1:8080

  # Windows / 开发机：本地加载基础模型
  python scripts/live_check.py --backend transformers --model "E:/.../Qwen3-VL-2B-Instruct"

  # 有微调 adapter 时
  python scripts/live_check.py --backend transformers --adapter runs/M_final/adapter

按键：
  空格 / Enter  拍照并推理一次
  1-5           切换对应类别是否在本次清单内
  A             清单恢复为五类全选
  R             清除上一次结果
  Q / Esc       退出
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from qwen_vlm import PROJECT_ROOT
    from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATE_NAME_ZH, normalize_required_items
    from qwen_vlm.common.derive import derive
    from qwen_vlm.inference.llamacpp_runner import PIXEL_TIERS
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen_vlm import PROJECT_ROOT
    from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH, STATE_NAME_ZH, normalize_required_items
    from qwen_vlm.common.derive import derive
    from qwen_vlm.inference.llamacpp_runner import PIXEL_TIERS

WINDOW_NAME = "Qwen3-VL Live Equipment Check"
DEFAULT_QUESTION = "按照本次实验清单，还缺少什么器材？"

DECISION_ZH = {"complete": "齐全", "incomplete": "不齐全", "unknown": "无法确认"}
DECISION_EN = {"complete": "COMPLETE", "incomplete": "INCOMPLETE", "unknown": "UNKNOWN"}
DECISION_RGB = {
    "complete": (80, 205, 125),
    "incomplete": (242, 100, 100),
    "unknown": (245, 190, 80),
    None: (242, 100, 100),
}
STATE_RGB = {
    "present": (120, 220, 150),
    "absent": (245, 130, 130),
    "unknown": (245, 200, 100),
}
NAME_EN = {
    "screwdriver": "screwdriver",
    "pliers": "pliers",
    "cable_bundle": "cable",
    "multimeter": "multimeter",
    "hammer": "hammer",
}
WHITE = (238, 242, 248)
GREY = (170, 178, 190)
DARK_PANEL = (10, 14, 20, 190)
DARK_BAR = (12, 17, 24, 205)

FONT_CANDIDATES = [
    os.environ.get("QWEN_VLM_FONT", ""),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def calm_stdio() -> None:
    """Windows 终端编码不一定是 UTF-8，避免中文 print 直接崩掉。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.encoding and stream.encoding.lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def find_font(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"字体文件不存在：{path}")
        return path
    for candidate in FONT_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def parse_camera_source(value: str):
    """'0' -> 0；'/dev/video2' 或 'http://...' 原样返回。"""
    value = str(value).strip()
    return int(value) if value.isdigit() else value


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def resolve_model_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("QWEN_VLM_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(PROJECT_ROOT / "artifacts/base/Qwen3-VL-2B-Instruct")
    for parent in PROJECT_ROOT.parents:
        candidates.append(parent / "PretrainedWeights/Qwen3-VL-2B-Instruct")
        candidates.append(parent / "Qwen3-VL-2B-Instruct")
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "找不到 Qwen3-VL-2B 模型目录（需要包含 config.json）。已查找：\n  "
        + searched
        + "\n请用 --model 指定，或设置环境变量 QWEN_VLM_MODEL_DIR。"
    )


def server_alive(url: str, timeout: float = 1.5) -> bool:
    try:
        import requests

        response = requests.get(url.rstrip("/") + "/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------


class CameraStream:
    """后台线程持续读帧，主线程随时取最新一帧。断线会自动重连。"""

    def __init__(self, source, width: int, height: int, fps: int):
        self.source = source
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.cap = None
        self.frame = None
        self.frame_clock = 0.0
        self.size: tuple[int, int] | None = None
        self.fps = 0.0
        self.error = "正在打开摄像头…"
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)
        # 线程还卡在 read() 时不要并发 release，交给线程自己的 finally。
        if not self.thread.is_alive() and self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open(self):
        apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_V4L2, cv2.CAP_ANY]
        for api in apis:
            cap = cv2.VideoCapture(self.source, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        raise RuntimeError(
            f"打不开摄像头 {self.source!r}。检查是否被其他程序占用；"
            "Windows 可试 --camera 1，Linux 可试 --camera /dev/video0。"
        )

    def _run(self) -> None:
        count = 0
        start = time.monotonic()
        failures = 0
        try:
            while not self.stop.is_set():
                if self.cap is None:
                    try:
                        self.cap = self._open()
                        with self.lock:
                            self.error = ""
                    except Exception as exc:
                        with self.lock:
                            self.error = str(exc)
                        self.stop.wait(1.0)
                        continue
                try:
                    ok, frame = self.cap.read()
                except cv2.error:
                    ok, frame = False, None
                now = time.monotonic()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 10:
                        failures = 0
                        self.cap.release()
                        self.cap = None
                        with self.lock:
                            self.error = "画面读取失败，正在重连…"
                        self.stop.wait(1.0)
                    else:
                        self.stop.wait(0.02)
                    continue
                failures = 0
                count += 1
                if now - start >= 1.0:
                    with self.lock:
                        self.fps = count / (now - start)
                    count = 0
                    start = now
                with self.lock:
                    self.frame = frame
                    self.frame_clock = now
                    self.size = (frame.shape[1], frame.shape[0])
                    self.error = ""
        except Exception as exc:  # noqa: BLE001 — 线程异常要显示到界面
            with self.lock:
                self.error = f"相机线程异常：{exc}"
        finally:
            if self.cap is not None:
                self.cap.release()

    def latest(self):
        """返回 (frame, fps, size, error)；超过 2.5 秒没有新帧时 frame 为 None。"""
        with self.lock:
            fresh = self.frame is not None and time.monotonic() - self.frame_clock <= 2.5
            return (
                self.frame if fresh else None,
                self.fps,
                self.size,
                self.error if not fresh else "",
            )


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


@dataclass
class LastResult:
    captured_at: datetime
    frame_bgr: np.ndarray
    decision: str | None
    answer_zh: str
    states: dict[str, str] | None
    required: list[str]
    missing: list[str]
    uncertain: list[str]
    timings: dict
    parse_status: str
    retry_count: int
    raw_text: str
    backend: str
    n_generated_tokens: int | None


class InferenceController:
    """模型加载、单请求推理和界面状态都在这里，界面只读快照。"""

    def __init__(self, question: str, save_dir: Path | None):
        self.lock = threading.RLock()
        self.question = question
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runner = None
        self.backend = ""
        self.model_state = "loading"  # loading / ready / error / off
        self.model_error = ""
        self.busy = False
        self.busy_started = 0.0
        self.message = ""
        self.message_until = 0.0
        self.last: LastResult | None = None

    # -- 状态 ---------------------------------------------------------------

    def note(self, text: str, seconds: float = 6.0) -> None:
        with self.lock:
            self.message = text
            self.message_until = time.monotonic() + seconds

    def set_off(self, text: str) -> None:
        with self.lock:
            self.model_state = "off"
            self.backend = "none"
            self.model_error = text

    def clear_result(self) -> None:
        with self.lock:
            self.last = None
            self.note("已清除上一次结果", 2.0)

    def ui_state(self) -> dict:
        with self.lock:
            return {
                "backend": self.backend,
                "model_state": self.model_state,
                "model_error": self.model_error,
                "busy": self.busy,
                "busy_s": time.monotonic() - self.busy_started if self.busy else 0.0,
                "message": self.message if time.monotonic() < self.message_until else "",
                "last": self.last,
            }

    # -- 加载 ---------------------------------------------------------------

    def load_async(self, factory) -> None:
        def run() -> None:
            try:
                runner = factory()
                with self.lock:
                    self.runner = runner
                    self.backend = getattr(runner, "backend_name", "runner")
                    self.model_state = "ready"
                    self.model_error = ""
                print(f"模型已就绪：{getattr(runner, 'backend_name', 'runner')}", flush=True)
            except Exception as exc:  # noqa: BLE001 — 要显示给用户
                traceback.print_exc()
                with self.lock:
                    self.model_state = "error"
                    self.model_error = str(exc)
                print(f"模型加载失败：{exc}", flush=True)

        threading.Thread(target=run, name="model-loader", daemon=True).start()

    # -- 推理 ---------------------------------------------------------------

    def submit(self, frame_bgr: np.ndarray, required: list[str]) -> None:
        with self.lock:
            if self.runner is None:
                if self.model_state == "loading":
                    self.note("模型还在加载，请稍候再按空格")
                elif self.model_state == "off":
                    self.note("当前是 --backend none，只预览画面，不推理")
                else:
                    self.note("模型没有加载成功，请看终端或界面上的错误")
                return
            if self.busy:
                self.note("上一次推理还没结束，请稍候")
                return
            self.busy = True
            self.busy_started = time.monotonic()
            self.message = ""
            runner = self.runner
            question = self.question
        thread = threading.Thread(
            target=self._work,
            args=(runner, frame_bgr.copy(), list(required), question),
            name="inference",
            daemon=True,
        )
        thread.start()

    def _work(self, runner, frame_bgr: np.ndarray, required: list[str], question: str) -> None:
        started = time.monotonic()
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            result = runner.predict(image, question, required)
            states = result.item_states
            decision = None
            missing: list[str] = []
            uncertain: list[str] = []
            answer = "模型输出没有通过解析，未推导结论。"
            if states is not None:
                try:
                    derived = derive(states, required)
                    decision = derived["decision"]
                    missing = derived["missing_required"]
                    uncertain = derived["uncertain_required"]
                    answer = derived["answer_zh"]
                except Exception as exc:  # noqa: BLE001
                    answer = f"状态后处理失败：{exc}"
                    states = None
            elapsed = max(time.monotonic() - started, float(result.timings.get("total_s") or 0.0))
            last = LastResult(
                captured_at=datetime.now(),
                frame_bgr=frame_bgr,
                decision=decision,
                answer_zh=answer,
                states=states,
                required=list(required),
                missing=list(missing),
                uncertain=list(uncertain),
                timings=dict(result.timings),
                parse_status=result.parse_status,
                retry_count=result.retry_count,
                raw_text=result.raw_text,
                backend=result.backend,
                n_generated_tokens=result.n_generated_tokens,
            )
            with self.lock:
                self.last = last
            self._print_result(last, elapsed)
            if self.save_dir is not None:
                self._save(last, elapsed)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                self.note(f"推理失败：{exc}", 10.0)
            print(f"推理失败：{exc}", flush=True)
        finally:
            with self.lock:
                self.busy = False

    @staticmethod
    def _print_result(last: LastResult, elapsed: float) -> None:
        print("\n" + "=" * 66, flush=True)
        print(f"[{last.captured_at:%H:%M:%S}] 结论：{DECISION_ZH.get(last.decision, '输出无法解析')}", flush=True)
        if last.states:
            line = "  ".join(f"{NAME_ZH[c]} {STATE_NAME_ZH.get(last.states[c], last.states[c])}" for c in CATEGORY_IDS)
            print("  " + line, flush=True)
        print(f"  {last.answer_zh}", flush=True)
        timing = last.timings.get("total_s") or elapsed
        print(
            f"  后端 {last.backend}，耗时 {float(timing):.2f} 秒，解析 {last.parse_status}，"
            f"重试 {last.retry_count}",
            flush=True,
        )
        if last.states is None:
            raw = last.raw_text.strip().replace("\n", " ")
            print(f"  原始输出：{raw[:500]}", flush=True)
        if last.missing or last.uncertain:
            print(
                "  清单内缺失："
                + ("、".join(NAME_ZH[c] for c in last.missing) or "无")
                + "；无法确认："
                + ("、".join(NAME_ZH[c] for c in last.uncertain) or "无"),
                flush=True,
            )
        print("=" * 66, flush=True)

    def _save(self, last: LastResult, elapsed: float) -> None:
        assert self.save_dir is not None
        stamp = f"{last.captured_at:%Y%m%d_%H%M%S_%f}"
        tag = last.decision or "parsefail"
        image_path = self.save_dir / f"live_{stamp}_{tag}.jpg"
        try:
            image_path.write_bytes(encode_jpeg(last.frame_bgr))
            payload = {
                "captured_at": last.captured_at.isoformat(timespec="seconds"),
                "backend": last.backend,
                "question": self.question,
                "required_items": last.required,
                "item_states": last.states,
                "decision": last.decision,
                "missing_required": last.missing,
                "uncertain_required": last.uncertain,
                "answer_zh": last.answer_zh,
                "parse_status": last.parse_status,
                "retry_count": last.retry_count,
                "n_generated_tokens": last.n_generated_tokens,
                "timings": last.timings,
                "elapsed_s": round(elapsed, 3),
                "raw_text": last.raw_text,
                "image": image_path.name,
                "image_size": [last.frame_bgr.shape[1], last.frame_bgr.shape[0]],
            }
            image_path.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  已保存：{image_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  保存失败：{exc}", flush=True)


# ---------------------------------------------------------------------------
# 画面渲染
# ---------------------------------------------------------------------------


class Renderer:
    """把 BGR 帧渲染成带中文/FPS/清单/结论的显示帧。没有中文字体时自动改用英文。"""

    def __init__(self, font_path: Path | None, max_display_width: int = 1280):
        self.font_path = font_path
        self.cjk = font_path is not None
        self.max_display_width = max_display_width
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def _font(self, size: int):
        size = max(11, int(size))
        if size not in self._fonts:
            if self.font_path is not None:
                self._fonts[size] = ImageFont.truetype(str(self.font_path), size)
            else:
                try:
                    self._fonts[size] = ImageFont.load_default(size=size)
                except TypeError:  # Pillow < 10
                    self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]

    def _zh(self, zh: str, en: str) -> str:
        return zh if self.cjk else en

    @staticmethod
    def _wrap(draw, text: str, font, max_width: int, max_lines: int | None = None) -> list[str]:
        lines: list[str] = []
        current = ""
        for ch in text:
            if ch == "\n":
                lines.append(current)
                current = ""
                continue
            if draw.textlength(current + ch, font=font) <= max_width:
                current += ch
            else:
                lines.append(current)
                current = ch
        if current or not lines:
            lines.append(current)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            lines[-1] = (last[:-1] + "…") if last else "…"
        return lines

    @staticmethod
    def _panel(draw, box, fill=DARK_PANEL, radius: int = 12) -> None:
        try:
            draw.rounded_rectangle(box, radius=radius, fill=fill)
        except AttributeError:
            draw.rectangle(box, fill=fill)

    def display_size(self, frame_bgr: np.ndarray) -> tuple[int, int]:
        h, w = frame_bgr.shape[:2]
        target = min(self.max_display_width, max(960, w))
        scale = target / w
        return int(round(w * scale)), int(round(h * scale))

    def render(self, frame_bgr: np.ndarray, ui: dict, required: list[str], fps: float,
               camera_size, camera_error: str) -> np.ndarray:
        width, height = self.display_size(frame_bgr)
        disp = frame_bgr if frame_bgr.shape[1] == width else cv2.resize(
            frame_bgr, (width, height), interpolation=cv2.INTER_AREA if frame_bgr.shape[1] > width else cv2.INTER_LINEAR
        )
        base = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        W, H = base.size
        s = W / 1280.0

        top_h = int(44 * s)
        check_h = int(56 * s)
        help_h = int(36 * s)

        # 顶栏
        draw.rectangle([0, 0, W, top_h], fill=DARK_BAR)
        title = self._zh("Qwen3-VL 实验室器材点检", "Qwen3-VL Lab Equipment Check")
        draw.text((int(26 * s), int(7 * s)), title, font=self._font(int(23 * s)), fill=WHITE)
        size_text = f"{camera_size[0]}x{camera_size[1]}" if camera_size else "no signal"
        backend = ui.get("backend") or "-"
        right = f"{backend} | {size_text} | {fps:4.1f} FPS"
        draw.text((W - int(14 * s), int(10 * s)), right, font=self._font(int(17 * s)), fill=GREY, anchor="ra")
        state_dot = {"ready": (70, 200, 110), "loading": (240, 190, 70), "error": (240, 90, 90), "off": (150, 150, 150)}
        draw.ellipse(
            [int(7 * s), int(16 * s), int(17 * s), int(26 * s)],
            fill=state_dot.get(ui.get("model_state"), (150, 150, 150)),
        )

        # 清单
        draw.rectangle([0, top_h, W, top_h + check_h], fill=(8, 12, 18, 150))
        x = int(14 * s)
        y = top_h + int(9 * s)
        label = self._zh("本次清单：", "Required: ")
        draw.text((x, y), label, font=self._font(int(19 * s)), fill=GREY)
        x += draw.textlength(label, font=self._font(int(19 * s)))
        req_font = self._font(int(20 * s))
        for cat in CATEGORY_IDS:
            on = cat in required
            mark = "[x]" if on else "[ ]"
            name = NAME_ZH[cat] if self.cjk else NAME_EN[cat]
            color = (120, 225, 150) if on else (150, 158, 170)
            text = f"{mark} {name}"
            draw.text((x, y), text, font=req_font, fill=color)
            x += draw.textlength(text, font=req_font) + int(18 * s)
        hint = self._zh("1-5 切换  A 全选", "1-5 toggle  A all")
        draw.text((W - int(14 * s), y + int(2 * s)), hint, font=self._font(int(15 * s)), fill=(120, 128, 140), anchor="ra")

        # 底部帮助
        draw.rectangle([0, H - help_h, W, H], fill=DARK_BAR)
        help_text = self._zh(
            "空格/Enter 拍照并推理   1-5 切换清单   A 全选   R 清除结果   Q/Esc 退出",
            "SPACE/Enter capture+infer   1-5 toggle list   A all   R clear   Q/Esc quit",
        )
        draw.text((W // 2, H - help_h // 2), help_text, font=self._font(int(16 * s)), fill=GREY, anchor="mm")

        # 相机错误
        if camera_error:
            err_font = self._font(int(17 * s))
            lines = self._wrap(draw, self._zh("相机：", "Camera: ") + camera_error, err_font, W - int(40 * s), 2)
            bar_h = int(10 * s) + len(lines) * int(24 * s)
            box = [int(10 * s), top_h + check_h + int(6 * s), W - int(10 * s), top_h + check_h + int(6 * s) + bar_h]
            self._panel(draw, box, fill=(120, 20, 20, 200))
            ty = box[1] + int(5 * s)
            for line in lines:
                draw.text((int(20 * s), ty), line, font=err_font, fill=(255, 220, 220))
                ty += int(24 * s)

        # 模型状态横幅
        if ui.get("model_state") == "loading":
            msg = self._zh("正在加载模型… 首次启动需要几十秒，画面可以继续预览", "Loading model… first start can take a while")
            self._center_banner(draw, W, int(H * 0.16), msg, self._font(int(19 * s)), (255, 225, 140), s)
        elif ui.get("model_state") == "error":
            err_font = self._font(int(17 * s))
            lines = self._wrap(draw, self._zh("模型加载失败：", "Model load failed: ") + ui.get("model_error", ""),
                               err_font, int(W * 0.82), 4)
            box_h = int(24 * s) + len(lines) * int(24 * s)
            y0 = int(H * 0.14)
            self._panel(draw, [int(W * 0.09), y0, int(W * 0.91), y0 + box_h], fill=(100, 20, 20, 215))
            ty = y0 + int(10 * s)
            for line in lines:
                draw.text((int(W * 0.11), ty), line, font=err_font, fill=(255, 220, 220))
                ty += int(24 * s)

        # 推理中
        if ui.get("busy"):
            txt = self._zh(f"正在推理… {ui.get('busy_s', 0):.1f} 秒", f"Inferring… {ui.get('busy_s', 0):.1f}s")
            self._center_banner(draw, W, int(H * 0.46), txt, self._font(int(26 * s)), (255, 235, 160), s, strong=True)

        # 上一次结果
        last: LastResult | None = ui.get("last")
        if last is not None:
            self._draw_result(draw, last, W, H, s, help_h)
            self._draw_thumbnail(base, draw, last, W, s, top_h + check_h)

        # 短消息
        if ui.get("message"):
            msg = ui["message"]
            font = self._font(int(18 * s))
            lines = self._wrap(draw, msg, font, int(W * 0.7), 2)
            box_h = int(16 * s) + len(lines) * int(24 * s)
            y0 = H - help_h - box_h - int(12 * s)
            self._panel(draw, [int(W * 0.15), y0, int(W * 0.85), y0 + box_h], fill=(30, 36, 46, 220))
            ty = y0 + int(8 * s)
            for line in lines:
                draw.text((W // 2, ty), line, font=font, fill=(235, 238, 245), anchor="ma")
                ty += int(24 * s)

        return cv2.cvtColor(np.array(Image.alpha_composite(base, overlay).convert("RGB")), cv2.COLOR_RGB2BGR)

    def _center_banner(self, draw, W: int, cy: int, text: str, font, color, s: float, strong: bool = False) -> None:
        tw = draw.textlength(text, font=font)
        pad_x, pad_y = int(22 * s), int(12 * s)
        box = [W / 2 - tw / 2 - pad_x, cy - pad_y, W / 2 + tw / 2 + pad_x, cy + pad_y]
        self._panel(draw, box, fill=(10, 14, 20, 220) if strong else (10, 14, 20, 180))
        draw.text((W / 2, cy), text, font=font, fill=color, anchor="mm")

    def _draw_result(self, draw, last: LastResult, W: int, H: int, s: float, help_h: int) -> None:
        pad = int(18 * s)
        panel_w = int(W * 0.65)
        x0 = int(14 * s)
        decision_font = self._font(int(32 * s))
        small_font = self._font(int(15 * s))
        answer_font = self._font(int(19 * s))
        state_font = self._font(int(18 * s))

        decision_color = DECISION_RGB.get(last.decision, DECISION_RGB[None])
        if last.decision:
            decision_text = self._zh("结论：" + DECISION_ZH[last.decision], DECISION_EN[last.decision])
        else:
            decision_text = self._zh("结论：模型输出无法解析", "DECISION: UNPARSEABLE")

        answer_lines = self._wrap(draw, last.answer_zh, answer_font, panel_w - 2 * pad, 3)
        state_rows: list[list[tuple[str, tuple, float]]] = []
        if last.states:
            row: list[tuple[str, tuple, float]] = []
            row_w = 0.0
            for cat in CATEGORY_IDS:
                state = last.states.get(cat, "unknown")
                name = NAME_ZH[cat] if self.cjk else NAME_EN[cat]
                text = f"{name} {STATE_NAME_ZH.get(state, state) if self.cjk else state}"
                width = draw.textlength(text, font=state_font)
                if row and row_w + width > panel_w - 2 * pad:
                    state_rows.append(row)
                    row, row_w = [], 0.0
                row.append((text, STATE_RGB.get(state, GREY), width))
                row_w += width + int(18 * s)
            if row:
                state_rows.append(row)

        title_h = int(42 * s)
        body_h = len(answer_lines) * int(26 * s) + int(6 * s)
        states_h = len(state_rows) * int(26 * s)
        panel_h = pad * 2 + title_h + body_h + states_h
        y0 = H - help_h - int(12 * s) - panel_h
        box = [x0, y0, x0 + panel_w, y0 + panel_h]
        self._panel(draw, box, fill=DARK_PANEL)

        ty = y0 + pad
        draw.text((x0 + pad, ty), decision_text, font=decision_font, fill=decision_color)
        meta = f"{last.captured_at:%H:%M:%S} · {float(last.timings.get('total_s') or 0.0):.2f}s"
        draw.text((x0 + panel_w - pad, ty + int(12 * s)), meta, font=small_font, fill=GREY, anchor="ra")
        ty += title_h
        for line in answer_lines:
            draw.text((x0 + pad, ty), line, font=answer_font, fill=WHITE)
            ty += int(26 * s)
        ty += int(6 * s)
        for row in state_rows:
            tx = x0 + pad
            for text, color, width in row:
                draw.text((tx, ty), text, font=state_font, fill=color)
                tx += width + int(18 * s)
            ty += int(26 * s)

    def _draw_thumbnail(self, base, draw, last: LastResult, W: int, s: float, top: int) -> None:
        frame = last.frame_bgr
        if frame is None or frame.size == 0:
            return
        tw = int(W * 0.24)
        th = max(1, int(frame.shape[0] * tw / frame.shape[1]))
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        x0 = W - tw - int(14 * s)
        y0 = top + int(12 * s)
        base.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (x0, y0))
        color = DECISION_RGB.get(last.decision, DECISION_RGB[None])
        draw.rectangle([x0 - 2, y0 - 2, x0 + tw + 1, y0 + th + 1], outline=color + (255,), width=max(2, int(3 * s)))
        label = self._zh("送入模型的画面", "Frame sent to model")
        draw.text((x0, y0 + th + int(5 * s)), label, font=self._font(int(15 * s)), fill=GREY)


# ---------------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------------


def build_backend(args):
    """按顺序尝试 llama-server 和本机 Transformers。"""
    if args.backend in ("auto", "llamacpp"):
        from qwen_vlm.inference.llamacpp_runner import LlamaCppRunner

        url = args.server.rstrip("/")
        if server_alive(url):
            if args.adapter:
                if args.backend == "llamacpp":
                    raise RuntimeError(
                        "llama.cpp 不能动态加载 LoRA adapter，请先合并并导出 GGUF，"
                        "或改用 --backend transformers。"
                    )
                print(
                    "注意：连上了 llama-server，--adapter 不会生效；要用 adapter 请加 --backend transformers。",
                    flush=True,
                )
            print(f"已连接 llama-server：{url}", flush=True)
            return LlamaCppRunner(
                base_url=url,
                max_pixels=PIXEL_TIERS[args.pixels],
                max_new_tokens=args.max_new_tokens,
            )
        if args.backend == "llamacpp":
            raise RuntimeError(
                f"连不上 llama-server（{url}）。请先启动服务，或用 --server 指定地址；"
                "也可以改用 --backend transformers 在本机加载模型。"
            )
        print(f"没有发现 llama-server（{url}），改用本机 Transformers。", flush=True)

    try:
        model_dir = resolve_model_dir(args.model)
    except FileNotFoundError as exc:
        if args.backend == "auto":
            raise RuntimeError(
                "既连不上 llama-server，也没找到本地 Qwen3-VL 模型。请先启动 llama-server"
                f"（--server {args.server}），或用 --model 指定本地模型目录。"
            ) from exc
        raise
    print(f"加载本地模型：{model_dir}", flush=True)
    import torch
    from qwen_vlm.inference.transformers_runner import TransformersRunner

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bf16" if device == "cuda" else "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    if device == "cpu" and dtype_name in ("bf16", "fp16"):
        print("CPU 上用 float32 更稳，已自动切换。", flush=True)
        dtype, dtype_name = torch.float32, "fp32"
    print(f"设备 {device}，dtype {dtype_name}，像素档 {args.pixels}", flush=True)
    runner = TransformersRunner(
        model_dir=model_dir,
        max_pixels=PIXEL_TIERS[args.pixels],
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        device=device,
    )
    if args.adapter:
        from peft import PeftModel

        adapter = Path(args.adapter).expanduser()
        if not adapter.is_absolute():
            adapter = (PROJECT_ROOT / adapter).resolve()
        print(f"加载 LoRA adapter：{adapter}", flush=True)
        runner.model = PeftModel.from_pretrained(runner.model, adapter).eval()
    return runner


# ---------------------------------------------------------------------------
# 自检 / 主循环
# ---------------------------------------------------------------------------


def self_test(args) -> int:
    """不加载模型，用一帧真实相机（没有就用合成图）渲染界面并保存，检查字体与布局。"""
    font = find_font(args.font)
    renderer = Renderer(font, max_display_width=args.display_width)
    frame = None
    cap = None
    try:
        cap = cv2.VideoCapture(parse_camera_source(args.camera), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
        if cap.isOpened():
            for _ in range(5):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
                    break
    except Exception:
        frame = None
    finally:
        if cap is not None:
            cap.release()
    source = "camera" if frame is not None else "synthetic"
    if frame is None:
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[:] = (38, 30, 28)
        for x in range(0, 1280, 80):
            cv2.line(frame, (x, 0), (x, 720), (60, 50, 46), 1)
        for y in range(0, 720, 60):
            cv2.line(frame, (0, y), (1280, y), (60, 50, 46), 1)
        cv2.rectangle(frame, (180, 250), (520, 470), (90, 120, 150), -1)
        cv2.rectangle(frame, (580, 280), (760, 430), (60, 140, 90), -1)
        cv2.putText(frame, "SYNTHETIC SELF-TEST FRAME", (180, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 210, 220), 2)

    states = {
        "screwdriver": "present",
        "pliers": "present",
        "cable_bundle": "present",
        "multimeter": "absent",
        "hammer": "present",
    }
    fake = LastResult(
        captured_at=datetime.now(),
        frame_bgr=frame.copy(),
        decision="incomplete",
        answer_zh="缺少万用表。",
        states=states,
        required=list(CATEGORY_IDS),
        missing=["multimeter"],
        uncertain=[],
        timings={"total_s": 1.82},
        parse_status="ok",
        retry_count=0,
        raw_text='{"item_states": {"screwdriver": "present", "pliers": "present", '
                 '"cable_bundle": "present", "multimeter": "absent", "hammer": "present"}}',
        backend="self-test",
        n_generated_tokens=42,
    )
    ui = {
        "backend": "self-test",
        "model_state": "ready",
        "model_error": "",
        "busy": False,
        "busy_s": 0.0,
        "message": "",
        "last": fake,
    }
    rendered = renderer.render(frame, ui, list(CATEGORY_IDS), fps=28.6, camera_size=(frame.shape[1], frame.shape[0]), camera_error="")
    out_dir = PROJECT_ROOT / "runs/live_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"self_test_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    out_path.write_bytes(encode_jpeg(rendered, 92))
    print(f"自检画面（来源：{source}）已保存：{out_path}", flush=True)
    if font is None:
        print("没有找到中文字体，界面使用英文。可用 --font 指定字体文件。", flush=True)
    else:
        print(f"中文字体：{font}", flush=True)
    return 0


def run_loop(args, camera: CameraStream, controller: InferenceController, renderer: Renderer,
             required: list[str]) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    placeholder = np.full((720, 1280, 3), 26, np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (420, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (190, 200, 210), 2)
    last_window_size = None
    started = time.monotonic()
    try:
        while True:
            frame, fps, size, camera_error = camera.latest()
            ui = controller.ui_state()
            if frame is None:
                display = renderer.render(placeholder, ui, required, fps, size, camera_error or "正在等待相机画面…")
            else:
                display = renderer.render(frame, ui, required, fps, size, camera_error)
            if display.shape[1::-1] != last_window_size:
                last_window_size = display.shape[1::-1]
                cv2.resizeWindow(WINDOW_NAME, last_window_size[0], last_window_size[1])
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (32, 13, 10):
                if frame is None:
                    controller.note("还没有相机画面，等画面出来再按空格")
                else:
                    controller.submit(frame, required)
            elif ord("1") <= key <= ord("5"):
                cat = CATEGORY_IDS[key - ord("1")]
                if cat in required:
                    required.remove(cat)
                else:
                    required.append(cat)
                required.sort(key=CATEGORY_IDS.index)
            elif key in (ord("a"), ord("A")):
                required[:] = list(CATEGORY_IDS)
            elif key in (ord("r"), ord("R")):
                controller.clear_result()
            if args.run_seconds and time.monotonic() - started >= args.run_seconds:
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    calm_stdio()
    parser = argparse.ArgumentParser(
        description="实时摄像头 + 空格触发一次 Qwen3-VL 器材完整性点检",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", default="0", help="摄像头编号（0）或 Linux 设备路径（/dev/video0）")
    parser.add_argument("--width", type=int, default=1280, help="请求的采集宽度")
    parser.add_argument("--height", type=int, default=720, help="请求的采集高度")
    parser.add_argument("--fps", type=int, default=30, help="请求的采集帧率")
    parser.add_argument("--backend", choices=["auto", "llamacpp", "transformers", "none"], default="auto",
                        help="auto：先连 llama-server，失败则本机加载；none：只预览不推理")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server 地址")
    parser.add_argument("--model", default=None, help="本机模型目录；默认自动查找 artifacts/base 或邻近的 PretrainedWeights")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录（仅 transformers 后端）")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--pixels", choices=sorted(PIXEL_TIERS), default="512", help="视觉像素档，384 更快，768 更清晰")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--items", nargs="*", default=None,
                        help="本次所需器材（中文名或类别 ID，空格分隔）；默认五类全部")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="送给模型的中文问题")
    parser.add_argument("--font", default=None, help="中文字体文件；默认自动查找微软雅黑/Noto CJK")
    parser.add_argument("--display-width", type=int, default=1280, help="窗口显示的最大宽度")
    parser.add_argument("--save-dir", default=None, help="每次推理保存画面和 JSON 的目录")
    parser.add_argument("--run-seconds", type=float, default=0.0, help="运行指定秒数后自动退出，0 表示一直运行")
    parser.add_argument("--self-test", action="store_true", help="只渲染一张自检画面并保存，不加载模型")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test(args)

    try:
        required = list(CATEGORY_IDS) if args.items is None else normalize_required_items(args.items)
    except Exception as exc:  # noqa: BLE001
        parser.error(f"{exc}；可用类别：{', '.join(CATEGORY_IDS)} 或中文名")

    try:
        font = find_font(args.font)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if font is None:
        print("警告：没有找到中文字体，界面会用英文。可以用 --font 指定，例如 C:/Windows/Fonts/msyh.ttc。", flush=True)

    print("实时器材点检", flush=True)
    print(f"  相机：{args.camera}（请求 {args.width}x{args.height} @ {args.fps}）", flush=True)
    if args.backend == "auto":
        print(f"  后端：auto（先试 {args.server}，失败则本机 Transformers）", flush=True)
    else:
        print(f"  后端：{args.backend}" + (f"（{args.server}）" if args.backend == "llamacpp" else ""), flush=True)
    print(f"  像素档：{args.pixels}（max_pixels={PIXEL_TIERS[args.pixels]}）", flush=True)
    print("  清单：" + ("、".join(NAME_ZH[c] for c in required) if required else "（空）"), flush=True)
    print("  按键：空格/Enter 推理，1-5 切换清单，A 全选，R 清除，Q/Esc 退出", flush=True)

    camera = CameraStream(parse_camera_source(args.camera), args.width, args.height, args.fps)
    controller = InferenceController(
        question=args.question,
        save_dir=Path(args.save_dir).expanduser() if args.save_dir else None,
    )
    renderer = Renderer(font, max_display_width=args.display_width)

    camera.start()
    if args.backend == "none":
        controller.set_off("只预览画面（--backend none）")
    else:
        controller.load_async(lambda: build_backend(args))

    try:
        run_loop(args, camera, controller, renderer, required)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
