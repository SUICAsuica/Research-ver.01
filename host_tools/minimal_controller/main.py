import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import websockets
from PIL import Image
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Camera & control settings
load_dotenv()

# Keep Hugging Face caches local to the project to avoid permission issues.
from pathlib import Path

HF_CACHE_BASE = Path(__file__).resolve().parents[2] / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_BASE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_BASE / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_BASE / "modules"))
os.environ.setdefault("TRANSFORMERS_DYNAMIC_MODULES_CACHE", str(HF_CACHE_BASE / "modules"))

CAM_URLS = [
    "http://192.168.4.1:8765/",
    "http://192.168.4.1:9000/mjpg",
    "http://192.168.4.1:9000/?action=stream",
]
WS_URL = "ws://192.168.4.1:8765/"
REGION_COUNT = 26  # Fields A..Z
JOYSTICK_REGION_INDEX = 10  # REGION_K
FRAME_RESIZE = (320, 240)
SLEEP_SECONDS = 0.05  # Keep <80 ms to hold MODE_APP_CONTROL
WARMUP_SECONDS = 5.0
ENABLE_PREVIEW = True
ENABLE_MASK_VIEW = False
LOG_LEVEL = logging.INFO
LOG_EVERY_N = 10

# Local VLM (smolVLM) settings
# Requires `mlx`, `mlx-vlm`, and a local copy of the smolVLM weights (install via `pip install mlx mlx-vlm`).
DEFAULT_SMOL_PATH = os.path.join("host_tools", "models", "smolvlm2-macos")
DEFAULT_SMOL_REPO = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
SMOL_MODEL_ID = os.getenv(
    "SMOL_MODEL_ID",
    DEFAULT_SMOL_PATH if os.path.isdir(DEFAULT_SMOL_PATH) else DEFAULT_SMOL_REPO,
)
SMOL_MAX_NEW_TOKENS = int(os.getenv("SMOL_MAX_NEW_TOKENS", "128"))
SMOL_TEMPERATURE = float(os.getenv("SMOL_TEMPERATURE", "0.0"))
SMOL_SYSTEM_PROMPT = os.getenv(
    "SMOL_SYSTEM_PROMPT",
    "You are a precise perception module for a robot car. Always answer in JSON.",
)
SMOL_USER_PROMPT = os.getenv(
    "SMOL_USER_PROMPT",
    (
        "Detect the primary white rectangular box (the car's target) in this image. "
        "Return a JSON object with fields: present (boolean), confidence (0-1), "
        "center_x, center_y, box_width, box_height (all 0-1 ratios of image width/height). "
        "If the box is missing, respond with present=false, confidence=0, and set the other values to 0."
    ),
)
VLM_INTERVAL_FRAMES = 10  # How often to refresh detection
VLM_MAX_AGE_SECONDS = 3.0  # Drop stale detections
STOP_AREA_THRESHOLD = 0.12  # Fraction of frame covered before stopping
TURN_GAIN = 90.0
MAX_FORWARD_SPEED = 80.0


@dataclass
class BoxDetection:
    center: Tuple[int, int]
    width_ratio: float
    height_ratio: float
    confidence: float

    @property
    def area_ratio(self) -> float:
        area = self.width_ratio * self.height_ratio
        return max(0.0, min(1.0, area))


def build_payload(x: int, y: int) -> str:
    fields = ["0"] * REGION_COUNT
    fields[JOYSTICK_REGION_INDEX] = f"{x},{y}"
    return "WS+" + ";".join(fields) + "\n"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_detection_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


class SmolVLMDetector:
    def __init__(self) -> None:
        from mlx_vlm.generate import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load as mlx_load

        self._generate = mlx_generate
        self._apply_chat_template = apply_chat_template
        self._model_id = SMOL_MODEL_ID
        logging.info("Loading smolVLM model %s", self._model_id)
        self.model, self.processor = mlx_load(
            self._model_id,
            trust_remote_code=True,
        )
        self.max_new_tokens = SMOL_MAX_NEW_TOKENS
        self.temperature = SMOL_TEMPERATURE

        messages = []
        if SMOL_SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SMOL_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": SMOL_USER_PROMPT})
        self.prompt = self._apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1,
            add_generation_prompt=True,
        )

    def detect(self, frame: np.ndarray) -> Optional[BoxDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        try:
            result = self._generate(
                self.model,
                self.processor,
                prompt=self.prompt,
                image=[image],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                verbose=False,
            )
        except Exception as exc:
            logging.warning("smolVLM generation failed: %s", exc)
            return None

        text = getattr(result, "text", "") if result else ""
        if not text:
            return None

        data = parse_detection_json(text)
        if not data:
            logging.debug("smolVLM raw output (unparsed): %s", text.strip())
            return None
        if not data.get("present"):
            return None

        try:
            width_ratio = clamp01(float(data.get("box_width", 0.0)))
            height_ratio = clamp01(float(data.get("box_height", 0.0)))
        except (TypeError, ValueError):
            return None

        if width_ratio == 0 or height_ratio == 0:
            return None

        try:
            confidence = clamp01(float(data.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        try:
            center_x = clamp01(float(data.get("center_x", 0.5)))
            center_y = clamp01(float(data.get("center_y", 0.5)))
        except (TypeError, ValueError):
            return None

        h, w = frame.shape[:2]
        cx_px = int(center_x * w)
        cy_px = int(center_y * h)
        return BoxDetection(
            center=(cx_px, cy_px),
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            confidence=confidence,
        )




_SMOL_DETECTOR: Optional[SmolVLMDetector] = None


def run_vlm_inference(frame: np.ndarray) -> Optional[BoxDetection]:
    global _SMOL_DETECTOR
    if _SMOL_DETECTOR is None:
        _SMOL_DETECTOR = SmolVLMDetector()
    return _SMOL_DETECTOR.detect(frame)


def detection_to_mask(shape: Tuple[int, int], detection: Optional[BoxDetection]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if detection is None:
        return mask
    h, w = shape
    box_w = int(clamp01(detection.width_ratio) * w)
    box_h = int(clamp01(detection.height_ratio) * h)
    x0 = int(np.clip(detection.center[0] - box_w / 2, 0, w - 1))
    x1 = int(np.clip(detection.center[0] + box_w / 2, 0, w - 1))
    y0 = int(np.clip(detection.center[1] - box_h / 2, 0, h - 1))
    y1 = int(np.clip(detection.center[1] + box_h / 2, 0, h - 1))
    mask[y0:y1, x0:x1] = 255
    return mask


def draw_detection(frame: np.ndarray, detection: Optional[BoxDetection]) -> None:
    if detection is None:
        return
    h, w = frame.shape[:2]
    box_w = int(clamp01(detection.width_ratio) * w / 2)
    box_h = int(clamp01(detection.height_ratio) * h / 2)
    cx, cy = detection.center
    top_left = (int(np.clip(cx - box_w, 0, w - 1)), int(np.clip(cy - box_h, 0, h - 1)))
    bottom_right = (int(np.clip(cx + box_w, 0, w - 1)), int(np.clip(cy + box_h, 0, h - 1)))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.circle(frame, detection.center, 6, (0, 255, 0), -1)
    cv2.putText(
        frame,
        f"conf={detection.confidence:.2f}",
        (top_left[0], max(20, top_left[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def decide_command(frame: np.ndarray, detection: Optional[BoxDetection]) -> Tuple[int, int, Optional[BoxDetection], np.ndarray]:
    h, w = frame.shape[:2]
    mask = detection_to_mask((h, w), detection)

    if detection is None:
        return 0, 0, None, mask

    area_ratio = detection.area_ratio
    if area_ratio >= STOP_AREA_THRESHOLD:
        return 0, 0, detection, mask

    cx, _ = detection.center
    dx = (cx - w / 2) / (w / 2)
    forward_scale = max(0.0, min(1.0, (STOP_AREA_THRESHOLD - area_ratio) / STOP_AREA_THRESHOLD))

    x_cmd = int(np.clip(dx * TURN_GAIN, -100, 100))
    y_cmd = int(np.clip(forward_scale * MAX_FORWARD_SPEED, 0, 100))
    return x_cmd, y_cmd, detection, mask


async def control_loop() -> None:
    cap: Optional[cv2.VideoCapture] = None
    for url in CAM_URLS:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            logging.info("Camera opened: %s", url)
            break
        cap.release()
        cap = None
    if cap is None:
        tried = ", ".join(CAM_URLS)
        raise RuntimeError(f"Failed to open camera; tried: {tried}")

    if WARMUP_SECONDS > 0:
        logging.info("Warming up camera for %.1f s", WARMUP_SECONDS)
        start = time.time()
        while time.time() - start < WARMUP_SECONDS:
            cap.read()
            time.sleep(0.05)

    if ENABLE_PREVIEW:
        logging.info("Preview window enabled")
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", FRAME_RESIZE[0], FRAME_RESIZE[1])
    if ENABLE_PREVIEW and ENABLE_MASK_VIEW:
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mask", FRAME_RESIZE[0], FRAME_RESIZE[1])

    running = True
    reconnect_delay = 2.0
    vlm_task: Optional[asyncio.Task] = None
    last_detection: Optional[BoxDetection] = None
    last_detection_time = 0.0

    try:
        while running:
            try:
                async with websockets.connect(WS_URL, ping_interval=10, ping_timeout=10) as ws:
                    logging.info("WebSocket connected: %s", WS_URL)
                    frame_count = 0
                    while running:
                        ok, frame = cap.read()
                        if not ok:
                            await asyncio.sleep(SLEEP_SECONDS)
                            continue

                        frame = cv2.resize(frame, FRAME_RESIZE)
                        frame_count += 1
                        now = time.time()

                        if last_detection is not None and (now - last_detection_time) > VLM_MAX_AGE_SECONDS:
                            last_detection = None

                        if vlm_task and vlm_task.done():
                            try:
                                result = vlm_task.result()
                            except Exception as exc:
                                logging.warning("VLM task failed: %s", exc)
                                result = None
                            last_detection = result
                            last_detection_time = time.time()
                            vlm_task = None

                        should_launch_vlm = (
                            vlm_task is None
                            and (
                                last_detection is None
                                or frame_count % VLM_INTERVAL_FRAMES == 0
                            )
                        )

                        if should_launch_vlm:
                            frame_for_vlm = frame.copy()
                            vlm_task = asyncio.create_task(
                                asyncio.to_thread(run_vlm_inference, frame_for_vlm)
                            )

                        x_cmd, y_cmd, detection, mask = decide_command(frame, last_detection)

                        if frame_count % LOG_EVERY_N == 0:
                            if detection:
                                logging.info(
                                    "cmd=(%d,%d) center=%s area=%.3f conf=%.2f",
                                    x_cmd,
                                    y_cmd,
                                    detection.center,
                                    detection.area_ratio,
                                    detection.confidence,
                                )
                            else:
                                logging.info("cmd=(%d,%d) no target", x_cmd, y_cmd)

                        if ENABLE_PREVIEW:
                            preview = frame.copy()
                            draw_detection(preview, detection)
                            cv2.imshow("preview", preview)
                            if ENABLE_MASK_VIEW:
                                cv2.imshow("mask", mask)
                            if cv2.waitKey(1) & 0xFF == 27:
                                running = False
                                break

                        try:
                            await ws.send(build_payload(x_cmd, y_cmd))
                        except (ConnectionClosedError, ConnectionClosedOK, TimeoutError) as exc:
                            logging.warning("WebSocket send failed: %s", exc)
                            break

                        await asyncio.sleep(SLEEP_SECONDS)
            except (ConnectionClosedError, ConnectionClosedOK, TimeoutError) as exc:
                logging.warning("WebSocket connection closed: %s", exc)
            except Exception as exc:
                logging.warning("WebSocket error: %s", exc)

            if not running:
                break
            logging.info("Reconnecting WebSocket after %.1f s", reconnect_delay)
            await asyncio.sleep(reconnect_delay)
    finally:
        if vlm_task and not vlm_task.done():
            vlm_task.cancel()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(control_loop())


if __name__ == "__main__":
    main()
