import asyncio
import contextlib
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
    "http://192.168.4.1:9000/mjpg",
]
WS_URL = "ws://192.168.4.1:8765/"
REGION_COUNT = 26  # Fields A..Z
JOYSTICK_REGION = 10  # REGION_K
HEAD_JOYSTICK_REGION = 16  # REGION_Q (pan/tilt stick stays centered)
MANUAL_MODE_REGION = 12  # REGION_M toggles app-control mode
FRAME_RESIZE = (320, 240)
SLEEP_SECONDS = 0.05  # Keep <80 ms to hold MODE_APP_CONTROL
WARMUP_SECONDS = 5.0
ENABLE_PREVIEW = True
ENABLE_MASK_VIEW = False
LOG_LEVEL = logging.DEBUG
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
    (
        "You are the perception core for a mobile robot. "
        "Always respond with strict JSON that matches the requested schema. "
        "If the white target rectangle is not visible or you are uncertain, "
        "set present=false and return 0 for all numeric values."
    ),
)
SMOL_USER_PROMPT = os.getenv(
    "SMOL_USER_PROMPT",
    (
        "Analyze the robot camera image and locate the white rectangular target (bright interior, dark border). "
        "Return JSON with fields: present (boolean), confidence (0-1), center_x, center_y, "
        "box_width, box_height (all normalized 0-1 relative to image width/height). "
        "Only use present=true when you can supply non-zero box_width and box_height and confidence >= 0.1. "
        "Example when visible: "
        '{"present": true, "confidence": 0.82, "center_x": 0.51, "center_y": 0.47, "box_width": 0.34, "box_height": 0.25}. '
        "Example when absent: "
        '{"present": false, "confidence": 0.0, "center_x": 0.0, "center_y": 0.0, "box_width": 0.0, "box_height": 0.0}. '
        "If the rectangle is missing or ambiguous, reuse the absent example exactly."
    ),
)
VLM_INTERVAL_FRAMES = 10  # How often to refresh detection
VLM_MAX_AGE_SECONDS = 3.0  # Drop stale detections
STOP_AREA_THRESHOLD = 0.12  # Fraction of frame covered before stopping
TURN_GAIN = 90.0
MAX_FORWARD_SPEED = 80.0
MIN_FORWARD_SPEED = 50  # Minimum joystick throttle once we decide to drive forward
HANDSHAKE_TIMEOUT = float(os.getenv("HANDSHAKE_TIMEOUT", "5.0"))
HANDSHAKE_CHECK = os.getenv("HANDSHAKE_CHECK", "SC")
HANDSHAKE_NAME = os.getenv("HANDSHAKE_NAME", "PC")
HANDSHAKE_TYPE = os.getenv("HANDSHAKE_TYPE", "Blank")


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
    xi = int(np.clip(x, -100, 100))
    yi = int(np.clip(y, -100, 100))

    fields = ["0"] * REGION_COUNT
    # Leave the mode-select switches (REGION_M..REGION_P) untouched so the UNO
    # firmware stays in MODE_APP_CONTROL. Setting them high flips the car into
    # autonomous scripts and ignores joystick data.
    fields[JOYSTICK_REGION] = f"{xi},{yi}"
    # Keep the second joystick (REGION_Q) centered so the ESP32 firmware
    # doesn't reuse stale head/tilt values from earlier sessions.
    fields[HEAD_JOYSTICK_REGION] = "0,0"
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
            text_attr = getattr(result, "text", None)
            logging.debug("smolVLM response: %r", text_attr)
        except Exception as exc:
            logging.warning("smolVLM generation failed: %s", exc)
            return None

        text = text_attr or ""
        if not text:
            logging.debug("smolVLM returned empty text.")
            return None

        data = parse_detection_json(text)
        if not data:
            logging.debug("smolVLM raw output (unparsed): %s", text.strip())
            return None
        if not data.get("present"):
            logging.debug("smolVLM indicates target absent.")
            return None

        try:
            confidence = clamp01(float(data.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        try:
            width_ratio = clamp01(float(data.get("box_width", 0.0)))
            height_ratio = clamp01(float(data.get("box_height", 0.0)))
        except (TypeError, ValueError):
            logging.debug("smolVLM returned invalid box dimensions: %s", data)
            return None

        if width_ratio <= 0.0 or height_ratio <= 0.0 or confidence <= 0.0:
            logging.debug("Discarding zero-sized or zero-confidence detection: %s", data)
            return None

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
    raw_forward = forward_scale * MAX_FORWARD_SPEED
    if raw_forward <= 0.0:
        y_cmd = 0
    else:
        y_cmd = int(np.clip(max(raw_forward, MIN_FORWARD_SPEED), 0, 100))
    return x_cmd, y_cmd, detection, mask


def build_handshake_payload(template: Optional[dict] = None) -> str:
    """Return the SunFounder controller greeting with fixed identity."""

    data = template.copy() if template else {}
    data.update(
        {
            "Check": HANDSHAKE_CHECK,
            "Name": HANDSHAKE_NAME,
            "Type": HANDSHAKE_TYPE,
        }
    )
    return json.dumps(data)


async def maybe_handle_check_frame(ws, text: str, send_lock: asyncio.Lock) -> bool:
    cleaned = text.strip()
    if not cleaned or not cleaned.startswith("{"):
        return False
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    check_value = data.get("Check")
    if not check_value:
        return False
    try:
        payload = build_handshake_payload(data)
        async with send_lock:
            await ws.send(payload)
        logging.info(
            "Responded to Check=%s with payload %s",
            check_value,
            payload,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to send handshake Ack: %s", exc)
        return False
    return True


async def wait_for_handshake(ws, send_lock: asyncio.Lock) -> None:
    logging.info("Waiting for ESP32 handshake...")
    start = time.time()
    while True:
        timeout = HANDSHAKE_TIMEOUT - (time.time() - start)
        if timeout <= 0:
            raise TimeoutError("Timed out waiting for ESP32 handshake response")
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Timed out waiting for ESP32 handshake message") from exc
        if isinstance(message, bytes):
            continue
        text = message.strip()
        if not text or text == "null" or text.startswith("pong"):
            continue
        logging.debug("WS recv text: %s", text)
        if await maybe_handle_check_frame(ws, text, send_lock):
            logging.info("ESP32 handshake complete")
            return


async def consume_server_messages(ws, send_lock: asyncio.Lock) -> None:
    try:
        async for message in ws:
            if isinstance(message, bytes):
                continue
            text = message.strip()
            if not text or text == "null" or text.startswith("pong"):
                continue
            await maybe_handle_check_frame(ws, text, send_lock)
    except (ConnectionClosedError, ConnectionClosedOK, TimeoutError):
        pass


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
                    send_lock = asyncio.Lock()
                    try:
                        await wait_for_handshake(ws, send_lock)
                    except TimeoutError as exc:
                        logging.error("Handshake with ESP32 failed: %s", exc)
                        await asyncio.sleep(reconnect_delay)
                        continue
                    receiver_task = asyncio.create_task(consume_server_messages(ws, send_lock))
                    frame_count = 0
                    try:
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
                                payload = build_payload(x_cmd, y_cmd)
                                logging.debug("WS payload -> %s", payload.strip())
                                async with send_lock:
                                    await ws.send(payload)
                            except (ConnectionClosedError, ConnectionClosedOK, TimeoutError) as exc:
                                logging.warning("WebSocket send failed: %s", exc)
                                break

                            await asyncio.sleep(SLEEP_SECONDS)
                    finally:
                        receiver_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await receiver_task
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
