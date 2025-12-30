import asyncio
import numpy as np
import os

import logging
from typing import AsyncGenerator
import time
import torch

from faster_whisper import WhisperModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper STT Server",
    description="FastAPI server for Whisper speech-to-text processing",
    version="1.0.0",
)


def get_optimal_device_and_compute_type():
    """Determine the best device and compute type for the system"""
    if torch.cuda.is_available():
        return "cuda", "float16"
    else:
        return "cpu", "int8"


device, compute_type = get_optimal_device_and_compute_type()
logger.info(f"Using device: {device}, compute_type: {compute_type}")


model = WhisperModel(
    model_size_or_path=os.path.join(os.path.dirname(__file__), "whisper_model"),
    device=device,
    compute_type=compute_type,
    # num_workers=1,
    # model_size_or_path="./whisper_model",
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
        }
    except Exception as e:
        logger.info(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "Whisper STT Server",
    }


async def run_stt(audio: bytes, language: str = "en") -> AsyncGenerator[str, None]:
    """Optimized Whisper STT with performance tweaks"""
    if not model:
        logger.info("Whisper model not available")
        yield "Whisper model not available"
        return

    # PCM int16 -> float32 in [-1, 1]
    audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    # if len(audio_float) < 1600:  # Less than 0.1 seconds at 16kHz
    #     return

    try:
        segments, _ = await asyncio.to_thread(
            model.transcribe,
            audio_float,
            language=language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False,
        )

        text = ""
        for segment in segments:
            text += f"{segment.text} "

        logger.info(f"text {text}")

        if text.strip():
            yield text.strip()

    except Exception as e:
        logger.info(f"Transcription error: {e}")
        yield str(e)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for STT processing"""
    await websocket.accept()
    logger.info("New WebSocket connection established")

    try:
        while True:
            # Receive audio data
            message = await websocket.receive_bytes()
            start_time = time.perf_counter()

            try:
                async for frame in run_stt(message):
                    print(f"Sending frame: {frame}")
                    await websocket.send_text(frame)

                processing_time = time.perf_counter() - start_time
                logger.info(f"Total processing time: {processing_time:.3f} seconds")

            except Exception as e:
                logger.info(f"Handler error: {e}")
                await websocket.send_text(f"ERROR: {str(e)}")

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.info(f"WebSocket handler error: {e}")


@app.on_event("startup")
async def startup_event():
    """Pre-warm the model on startup"""
    logger.info("Pre-warming model...")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    try:
        list(model.transcribe(dummy_audio, beam_size=1))
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9800, log_level="info", reload=False)
