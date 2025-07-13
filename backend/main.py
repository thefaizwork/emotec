"""
FastAPI server that runs facial and audio emotion models concurrently in real-time.
It exposes two endpoints:
POST /combine/start  – begins a background thread that captures webcam frames and microphone audio,
                       runs the pre-trained models, and stores the streamed predictions.
POST /combine/stop   – stops the thread and returns a simple summary of the emotions
                       detected separately for facial and audio channels.

NOTE:
* This is a minimal working prototype. Real-world production requires
  more sophisticated audio buffering, error handling, and security.
* The facial model is expected to accept a single grayscale face image
  of shape (48, 48, 1) scaled to [0,1]. Adjust `preprocess_face()` if
  your model expects something different.
* The audio model is expected to accept a log-Mel-spectrogram (96×64)
  or similar 2-D time-frequency representation. Adjust
  `preprocess_audio_chunk()` accordingly.
"""

# ---------------------------------------------------------------------------
# Compatibility shim – Python 3.12 removed distutils; many deps still need it
# ---------------------------------------------------------------------------
import sys, importlib
try:
    import distutils  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import setuptools  # ensures vendor copy is present
    # Alias vendor `_distutils` as standard module so imports like 'distutils.util' work
    distutils = importlib.import_module('setuptools._distutils')
    sys.modules['distutils'] = distutils

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict
import uuid
from datetime import datetime

import cv2
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "saved_models"
FACIAL_MODEL_PATH = MODEL_DIR / "facial_emotion_model.h5"
AUDIO_MODEL_PATH = MODEL_DIR / "audio_emotion_model.h5"
AUDIO_LABELS_PATH = MODEL_DIR / "audio_emotion_labels.npy"

# Facial emotion classes (adjust to match your own model)
FACIAL_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# ---------------------------------------------------------------------------
# Load models once at start-up
# ---------------------------------------------------------------------------
try:
    facial_model = tf.keras.models.load_model(FACIAL_MODEL_PATH, compile=False)
except Exception as exc:
    raise RuntimeError(f"Failed to load facial model: {exc}")

try:
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
except Exception as exc:
    raise RuntimeError(f"Failed to load audio model: {exc}")

try:
    AUDIO_LABELS = np.load(AUDIO_LABELS_PATH, allow_pickle=True).tolist()
except Exception as exc:
    raise RuntimeError(f"Failed to load audio labels: {exc}")

# ---------------------------------------------------------------------------
# Globals to manage the background analysis
# ---------------------------------------------------------------------------
app = FastAPI(title="Combined Facial + Audio Emotion API")

# Enable CORS so that the Next.js frontend (http://localhost:3000) can call the API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://emotec.onrender.com",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("emotion_api")

_analysis_lock = threading.Lock()
_is_running: bool = False
_thread: threading.Thread | None = None

_facial_preds: list[str] = []
_audio_preds: list[str] = []

# Determine audio model expected input format
AUDIO_INPUT_RANK = len(audio_model.input_shape)
EXPECTED_FLAT_LEN: int | None = audio_model.input_shape[1] if AUDIO_INPUT_RANK == 2 else None
EXPECTED_CNN_SHAPE: tuple[int,int] | None = (
    (audio_model.input_shape[1] or 96,
     audio_model.input_shape[2] or 64)
) if AUDIO_INPUT_RANK != 2 else None

# Time-series storage: (timestamp_seconds, emotion)
_facial_time_series: list[tuple[float, str]] = []
_audio_time_series: list[tuple[float, str]] = []

_sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helper / preprocessing functions
# ---------------------------------------------------------------------------

def preprocess_face(frame: np.ndarray) -> np.ndarray:
    """Detect face, crop, resize to 48×48 (grayscale) and scale to [0,1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))  # tweak if model expects another size
    face = face / 255.0  # scale
    face = np.expand_dims(face, axis=(0, -1)).astype("float32")  # (1, 48, 48, 1)
    return face


def preprocess_audio_chunk(raw: np.ndarray, sr: int) -> np.ndarray:
    """Convert raw mono audio chunk into the correct input tensor for the audio model.

    Handles two model types:
    1. CNN-like model expecting shape (1, H, W, 1)
    2. Dense model expecting flat vector (1, N)
    The function automatically inspects `audio_model.input_shape` to choose the path.
    """

    # If stereo convert to mono
    if raw.ndim > 1:
        raw = np.mean(raw, axis=1)

    # 1. Build log-Mel spectrogram
    mel = librosa.feature.melspectrogram(y=raw, sr=sr, n_mels=64, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Ensure consistent time×freq resolution for CNN input
    target_height, target_width = (EXPECTED_CNN_SHAPE or (96, 64))
    log_mel = cv2.resize(log_mel, (target_width, target_height))  # cv2 expects (w, h)

    # Normalise rough dB range to ~[0,1]
    log_mel = (log_mel + 40) / 40

    # Dense model → flatten/pad/truncate
    if EXPECTED_FLAT_LEN is not None:
        flat = log_mel.flatten()
        if flat.size < EXPECTED_FLAT_LEN:
            flat = np.pad(flat, (0, EXPECTED_FLAT_LEN - flat.size))
        if flat.size > EXPECTED_FLAT_LEN:
            flat = flat[:EXPECTED_FLAT_LEN]
        return np.expand_dims(flat.astype("float32"), 0)  # (1, N)

    # CNN model → add channel dim
    return np.expand_dims(log_mel.astype("float32"), (0, -1))  # (1, H, W, 1)

# ---------------------------------------------------------------------------
# Helper functions for report generation
# ---------------------------------------------------------------------------

def compress_series(series: list[tuple[float, str]]) -> list[dict]:
    """Compress consecutive identical emotions into segments with start/end times."""
    if not series:
        return []
    segments = []
    cur_emo = series[0][1]
    seg_start = series[0][0]
    for ts, emo in series[1:]:
        if emo != cur_emo:
            segments.append({"emotion": cur_emo, "start": round(seg_start, 2), "end": round(ts, 2)})
            cur_emo = emo
            seg_start = ts
    segments.append({"emotion": cur_emo, "start": round(seg_start, 2), "end": round(series[-1][0], 2)})
    return segments


EMO_TIPS = {
        "Angry": "Consider short breaks or mindfulness exercises to cool down.",
        "Disgust": "Try focusing on positive thoughts to soften facial tension.",
        "Fear": "Deep-breathing can help reduce anxious expressions and tone.",
        "Sad": "Think about an uplifting memory or smile gently to lift mood.",
        "Happy": "Great! Keep that positive energy going.",
        "Surprise": "Maintain calm eye contact to avoid looking startled.",
        "Calm": "Excellent composure – keep it up!",
        "Neutral": "Neutral is fine; sprinkle in a smile now and then.",
    }


def build_suggestions(facial_counts: Counter, audio_counts: Counter) -> list[str]:
    """Build per-emotion suggestions for emotions that actually occurred."""
    combined = set(facial_counts) | set(audio_counts)
    suggestions: list[str] = []
    for emo in combined:
        tip = EMO_TIPS.get(emo)
        if tip:
            # prefix channel depending on where emotion was dominant
            fac = facial_counts.get(emo, 0)
            aud = audio_counts.get(emo, 0)
            channel = "Facial" if fac >= aud else "Audio"
            suggestions.append(f"{channel}: {tip}")
    if not suggestions:
        suggestions.append("Great job! You maintained mostly positive emotions during the session.")
    return suggestions

# ---------------------------------------------------------------------------
# Background analysis thread
# ---------------------------------------------------------------------------

def _analysis_loop():
    """Background loop that captures webcam & mic and runs models.
    Ensures `_is_running` flag is reset if any exception bubbles out."""
    global _is_running
    video_cap = None
    try:
        # Open resources only inside the thread
        video_cap = cv2.VideoCapture(0)  # default webcam
        if not video_cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0). Is it in use by another app?")

        # Configure audio stream (mono, 16-kHz)
        sample_rate = 16000
        block_size = 2048  # ~128 ms

        start_ts = time.time()
        audio_queue: list[np.ndarray] = []

        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("SoundDevice status: %s", status)
            audio_queue.append(indata.copy())

        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=block_size,
            callback=_audio_callback,
        ):
            while _is_running:
                # Facial
                ret, frame = video_cap.read()
                if ret:
                    face_input = preprocess_face(frame)
                    facial_logits = facial_model.predict(face_input, verbose=0)[0]
                    facial_pred = FACIAL_LABELS[int(np.argmax(facial_logits))]
                    _facial_preds.append(facial_pred)
                    _facial_time_series.append((time.time() - start_ts, facial_pred))

                # Audio
                if audio_queue:
                    raw_block = audio_queue.pop(0).flatten()
                    audio_input = preprocess_audio_chunk(raw_block, sample_rate)
                    audio_logits = audio_model.predict(audio_input, verbose=0)[0]
                    audio_pred = AUDIO_LABELS[int(np.argmax(audio_logits))]
                    _audio_preds.append(audio_pred)
                    _audio_time_series.append((time.time() - start_ts, audio_pred))

                time.sleep(0.2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis loop crashed: %s", exc)
    finally:
        _is_running = False
        if video_cap is not None:
            video_cap.release()

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict/face")
async def predict_face(image: UploadFile = File(...)) -> Dict:
    """Predict facial emotion from an uploaded image (JPEG/PNG)."""
    try:
        data = await image.read()
        file_arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid image")
        emotion_input = preprocess_face(bgr)
        logits = facial_model.predict(emotion_input, verbose=0)[0]
        pred = FACIAL_LABELS[int(np.argmax(logits))]
        return {"emotion": pred}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/audio")
async def predict_audio(audio: UploadFile = File(...)) -> Dict:
    """Predict audio emotion from a ≤1-second mono 16-kHz WAV/WEBM chunk."""
    try:
        raw_bytes = await audio.read()
        import soundfile as sf, io
        data, sr = sf.read(io.BytesIO(raw_bytes))
        if sr != 16000:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        # ensure mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # keep max 1 sec
        if data.shape[0] > sr:
            data = data[-sr:]
        emotion_input = preprocess_audio_chunk(data.astype(np.float32), sr)
        logits = audio_model.predict(emotion_input, verbose=0)[0]
        pred = AUDIO_LABELS[int(np.argmax(logits))]
        return {"emotion": pred}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

# ---------------------------------------------------------------------------
# Legacy combined analysis endpoints (optional, still available)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.post("/combine/start")
async def start_combined_analysis() -> Dict[str, str]:
    """Start background combined facial + audio emotion analysis."""
    global _is_running, _thread
    with _analysis_lock:
        if _is_running:
            raise HTTPException(status_code=400, detail="Analysis already running")
        _is_running = True
        _thread = threading.Thread(target=_analysis_loop, daemon=True)
        _thread.start()
    return {"status": "started"}


@app.post("/combine/stop")
async def stop_combined_analysis() -> Dict:
    """Stop analysis and return emotion summaries."""
    global _is_running, _thread, _facial_preds, _audio_preds
    with _analysis_lock:
        if not _is_running:
            raise HTTPException(status_code=400, detail="Analysis not running")
        _is_running = False
    # Wait a bit for the thread to finish
    if _thread is not None:
        _thread.join(timeout=2)
        _thread = None

    facial_summary = Counter(_facial_preds)
    audio_summary = Counter(_audio_preds)

    facial_timeline = compress_series(_facial_time_series)
    audio_timeline = compress_series(_audio_time_series)
    suggestions = build_suggestions(facial_summary, audio_summary)

    result = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "facial_summary": dict(facial_summary),
        "audio_summary": dict(audio_summary),
        "facial_timeline": facial_timeline,
        "audio_timeline": audio_timeline,
        "suggestions": suggestions,
    }

    _sessions[result["id"]] = result

    # Reset for next session
    _facial_preds.clear()
    _audio_preds.clear()
    _facial_time_series.clear()
    _audio_time_series.clear()

    return result


@app.get("/sessions")
async def list_sessions() -> Dict:
    """List all recorded sessions."""
    return {"sessions": list(_sessions.values())}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "deleted", "id": session_id}


@app.get("/health")
async def health_status() -> Dict:
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "analysis_running": _is_running,
        "models": {
            "facial_loaded": True,
            "audio_loaded": True,
        },
    }

# ---------------------------------------------------------------------------
# Optional: run with `python main.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
