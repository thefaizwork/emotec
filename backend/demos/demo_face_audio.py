"""demo_face_audio.py

Standalone demonstration script that visualizes *both* facial and audio
emotion recognition in real time – without needing to start the FastAPI
backend.

Features
--------
1. Opens a webcam window with MediaPipe FaceMesh overlay and shows the
   current **facial emotion** label in the upper-left corner.
2. Captures microphone audio, classifies a 1-second rolling window every
   second, and prints the **audio emotion** to the console.
3. Command-line flags let you choose the camera index and audio device.
   Example:

    python demos/demo_face_audio.py --cam 1 --audio 3

Requirements
------------
```
opencv-python, mediapipe, tensorflow, librosa, sounddevice
```
All are already listed in `requirements.txt`.

Note: This script re-uses the same model-loading and preprocessing
helpers defined in `main.py` to avoid code duplication.
"""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path
from typing import List
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so we can import `main`
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import librosa
import mediapipe as mp
import numpy as np
import sounddevice as sd

# Re-use helpers / models from the FastAPI module
from main import (
    AUDIO_LABELS,
    FACIAL_LABELS,
    audio_model,
    facial_model,
    preprocess_audio_chunk,
    preprocess_face,
)

# ---------------------------------------------------------------------------
# MediaPipe face mesh setup
# ---------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Color for drawing face mesh
_DRAW_SPEC = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face+audio emotion demo")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--audio", type=int, default=None, help="SoundDevice input device index (see `sd.query_devices()`) ")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate (Hz)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("[Audio]", status, flush=True)
    # Flatten to mono float32
    audio_queue.put(indata.copy().flatten())


def audio_emotion_worker(sample_rate: int) -> None:
    """Consume 1-second chunks from audio_queue, classify, and print."""
    buf: List[np.ndarray] = []
    collected = 0
    target = sample_rate  # samples per second (mono)
    last_print = time.time()

    while True:
        try:
            block = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # Timed out → check exit flag later (handled in main loop)
            return
        buf.append(block)
        collected += len(block)
        if collected >= target:
            # Concatenate and keep only the last `target` samples
            data = np.concatenate(buf)[-target:]
            buf = [data]  # reset but keep remainder for overlap-free 1s windows
            collected = len(data)

            # Classify
            audio_input = preprocess_audio_chunk(data, sample_rate)
            logits = audio_model.predict(audio_input, verbose=0)[0]
            pred = AUDIO_LABELS[int(np.argmax(logits))]
            print(f"Audio emotion ({time.strftime('%H:%M:%S')}): {pred}")


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # ------------- Start audio stream -------------
    stream = sd.InputStream(
        device=args.audio,
        channels=1,
        samplerate=args.sr,
        callback=audio_callback,
        blocksize=2048,
    )
    stream.start()

    # Start a background thread for audio emotion processing
    import threading

    audio_thread = threading.Thread(target=audio_emotion_worker, args=(args.sr,), daemon=True)
    audio_thread.start()

    # ------------- Video capture -------------
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame – exiting.")
            break

        # Predict facial emotion every frame (~30fps) – cheap once model is loaded
        face_input = preprocess_face(frame)
        logits = facial_model.predict(face_input, verbose=0)[0]
        facial_pred = FACIAL_LABELS[int(np.argmax(logits))]

        # Overlay face mesh for visualization
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=_DRAW_SPEC,
                    connection_drawing_spec=_DRAW_SPEC,
                )

        # Put facial emotion label
        cv2.putText(
            frame,
            f"Facial: {facial_pred}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Face & Audio Emotion Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    main()
