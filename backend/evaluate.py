"""Evaluate saved facial & audio models on FER-2013 and RAVDESS datasets.

Usage:
    1. Copy FER-2013 images into datasets/fer2013/train/ and datasets/fer2013/test/ as per README.
    2. Copy RAVDESS wavs into datasets/ravdess/Actor_?? folders.
    3. Run `python evaluate.py` from the project root. It will print accuracy.

Edit paths or preprocessing below if your model expects something different.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Compatibility shim for Python 3.12 – many libs still expect `distutils`
# ---------------------------------------------------------------------------
try:
    import distutils  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Importing setuptools reintroduces a vendored copy of distutils
    import setuptools  # noqa: F401

import cv2
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
FER_DIR = DATASETS_DIR / "fer2013"            # train/ test/ subfolders
RAVDESS_DIR = DATASETS_DIR / "ravdess"        # Actor_01 … Actor_24

FACIAL_MODEL_PATH = PROJECT_ROOT / "saved_models/facial_emotion_model.h5"
AUDIO_MODEL_PATH = PROJECT_ROOT / "saved_models/audio_emotion_model.h5"

FER_IMG_SIZE = (48, 48)
# For spectrogram-based CNN audio models.
AUDIO_MEL_SHAPE_CNN = (96, 64)  # (time, freq)

# For flattened-dense audio models (e.g. 128×128 -> 16384).
AUDIO_FLAT_MELS = 128
AUDIO_FLAT_FRAMES = 128

FER_CLASSES_MASTER = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral",
]
RAVDESS_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}
# ---------------------------------------------------------------------------


def load_fer_split(split: str, allowed_indices: list[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, y for FER-2013 split ('train' or 'test')."""
    xs, ys = [], []
    for idx, cls in enumerate(FER_CLASSES_MASTER):
        for img_path in (FER_DIR / split / cls).glob("*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if idx not in allowed_indices:
                continue  # skip classes model cannot predict
            img = cv2.resize(img, FER_IMG_SIZE)
            xs.append(img.astype("float32") / 255.0)
            ys.append(allowed_indices.index(idx))  # reindex
    return np.expand_dims(np.array(xs), -1), np.array(ys)


def preprocess_audio(path: Path, flat_len: int | None = None, cnn_shape: tuple[int,int] | None = None) -> np.ndarray:
    y, sr = librosa.load(path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if flat_len is not None:
        # Build a square-ish 128×128 (assuming flat_len≈16384) matrix then flatten.
        target_side = int(flat_len ** 0.5)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_side, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Ensure exactly target_side frames
        import librosa.util as lutil
        log_mel = lutil.fix_length(log_mel, size=target_side, axis=1)
        log_mel = (log_mel + 40) / 40
        flat = log_mel.flatten()
        if flat.size < flat_len:
            flat = np.pad(flat, (0, flat_len - flat.size))
        else:
            flat = flat[:flat_len]
        return flat.astype("float32")
    else:
        target = cnn_shape if cnn_shape else AUDIO_MEL_SHAPE_CNN
        log_mel = cv2.resize(log_mel, target[::-1])
        log_mel = (log_mel + 40) / 40  # roughly 0-1
        return np.expand_dims(log_mel.astype("float32"), -1)


def filename_to_emotion(filename: str) -> str | None:
    try:
        code = filename.split("-")[2]
        return RAVDESS_MAP[code]
    except (IndexError, KeyError):
        return None


def load_ravdess(allowed_indices: list[int], flat_len: int | None, cnn_shape: tuple[int,int] | None) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for wav in RAVDESS_DIR.rglob("*.wav"):
        emo = filename_to_emotion(wav.name)
        if emo is None or emo not in FER_CLASSES_MASTER:
            continue
        label_idx = FER_CLASSES_MASTER.index(emo)
        if label_idx not in allowed_indices:
            continue
        xs.append(preprocess_audio(wav, flat_len, cnn_shape))
        ys.append(allowed_indices.index(label_idx))
    return np.array(xs), np.array(ys)


def evaluate(model_path: Path, x: np.ndarray, y: np.ndarray, title: str):
    print(f"\nEvaluating {title} …  (N={len(y)})")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    loss, acc = model.evaluate(x, y, batch_size=64, verbose=0)
    print(f"{title} accuracy: {acc * 100:.2f}%\n")


if __name__ == "__main__":
    if not FER_DIR.exists():
        print("⚠️  FER dataset not found, skipping.")
    if not RAVDESS_DIR.exists():
        print("⚠️  RAVDESS dataset not found, skipping.")

    # Detect number of classes from models
    facial_model = tf.keras.models.load_model(FACIAL_MODEL_PATH, compile=False)
    facial_classes = list(range(facial_model.output_shape[-1]))

    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
    audio_classes = list(range(audio_model.output_shape[-1]))

    # Determine if audio model expects flat vector or spectrogram
    audio_input_rank = len(audio_model.input_shape)
    if audio_input_rank == 2:
        expected_flat_len = audio_model.input_shape[1]
        expected_cnn_shape = None
    else:
        expected_flat_len = None  # CNN style
        expected_cnn_shape = (audio_model.input_shape[1] or AUDIO_MEL_SHAPE_CNN[0],
                              audio_model.input_shape[2] or AUDIO_MEL_SHAPE_CNN[1])

    # Facial model on FER test split
    if FER_DIR.exists():
        x_test, y_test = load_fer_split("test", facial_classes)
        evaluate(FACIAL_MODEL_PATH, x_test, y_test, f"Facial model (FER test – {len(facial_classes)} classes)")

    # Audio model on RAVDESS
    if RAVDESS_DIR.exists():
        x_audio, y_audio = load_ravdess(audio_classes, expected_flat_len, expected_cnn_shape)
        evaluate(AUDIO_MODEL_PATH, x_audio, y_audio, f"Audio model (RAVDESS – {len(audio_classes)} classes)")

        print("⚠️  Dataset folders not found. Copy files as described in datasets/*/README.md and re-run.")
        exit(1)

    # Facial model on FER test split
    x_test, y_test = load_fer_split("test")
    evaluate(FACIAL_MODEL_PATH, x_test, y_test, "Facial model (FER test)")

    # Audio model on RAVDESS
    x_audio, y_audio = load_ravdess()
    evaluate(AUDIO_MODEL_PATH, x_audio, y_audio, "Audio model (RAVDESS)")
