# Combined Facial & Audio Emotion Detection API

This FastAPI server exposes two simple endpoints to run your **pre-trained facial and audio emotion models** in real-time:

* `POST /combine/start` – starts a background thread that records webcam frames and microphone audio, sends them to the models, and logs predictions.
* `POST /combine/stop`  – stops the capture thread and returns a count of each emotion detected separately for the facial and audio streams.

## Running locally

```bash
# 1. (Optional) create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python main.py
```

The API will be available at `http://localhost:8000`. You can also visit `http://localhost:8000/docs` for the interactive Swagger UI.

## Folder structure
```
├── main.py                  ← FastAPI application
├── requirements.txt         ← Python dependencies
├── saved_models             ← Pre-trained models and label file
│   ├── facial_emotion_model.h5
│   ├── audio_emotion_model.h5
│   └── audio_emotion_labels.npy
└── README.md                ← This file
```

---
### Notes & Limitations

* This is a **minimal prototype**. You may need to adjust pre-processing (`preprocess_face`, `preprocess_audio_chunk`) to match exactly what your models expect.
* Audio capture uses `sounddevice` and may require PortAudio on your system.
* On Windows, you might need to install the Microsoft C++ Build Tools if you encounter compilation errors when installing dependencies.
* For production you would add error checking, authentication, richer summaries, database storage, etc.
