"""Entry point to run the FastAPI server locally.

Usage:
    python run.py            # starts the server on http://127.0.0.1:8000 with reload

After the server starts, open:
    * http://127.0.0.1:8000/health  – quick JSON health check
    * http://127.0.0.1:8000/docs    – Swagger UI to exercise /combine/start and /combine/stop
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="debug",
    )
