import os
import json
import time
import asyncio

import logging
import numpy as np
import torch
from typing import AsyncGenerator, Dict, Optional, List
from loguru import logger
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from text_processing.text_processor import smart_split
from kokoro import KModel, KPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kokoro")


class KokoroConfig:
    def __init__(
        self,
        voice: str = "af_bella",
        speed: float = 1.0,
        lang_code: str = "a",
        volume_multiplier: float = 1.0,
    ):
        self.voice = voice
        self.speed = max(0.1, min(5.0, speed))
        self.lang_code = lang_code
        self.volume_multiplier = max(0.1, min(5.0, volume_multiplier))


def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class KokoroState:
    def __init__(self):
        self.device = get_optimal_device()
        self.model: Optional[KModel] = None
        self.pipelines: Dict[str, KPipeline] = {}


state = KokoroState()

MODEL_DIR = "kokoro_model/"
VOICES_DIR = "kokoro_model/voices"
MODEL_FILE = "kokoro-v1_0.pth"
CONFIG_FILE = "config.json"


def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in state.pipelines:
        logger.info(f"Creating pipeline for {lang_code}")
        state.pipelines[lang_code] = KPipeline(
            lang_code=lang_code, model=state.model, device=state.device
        )
    return state.pipelines[lang_code]


async def load_model():
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    config_path = os.path.join(MODEL_DIR, CONFIG_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading Kokoro model on {state.device}")

    model = KModel(config=config_path, model=model_path).eval()

    if state.device == "cuda":
        model = model.cuda()
    elif state.device == "mps":
        model = model.to(torch.device("mps"))
    else:
        model = model.cpu()

    state.model = model

    logger.info("Model loaded successfully")


async def get_voice_path(voice_name: str) -> str:
    voice_path = os.path.join(VOICES_DIR, f"{voice_name}.pt")
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    return voice_path


async def create_audio_stream(
    text: str, tts_config: KokoroConfig
) -> AsyncGenerator[bytes, None]:
    voice_path = await get_voice_path(tts_config.voice)
    if not voice_path:
        raise ValueError(f"Voice file not found: {voice_path}")

    pipeline = get_pipeline(tts_config.lang_code)
    for result in pipeline(
        text,
        voice=voice_path,
        speed=tts_config.speed,
        model=state.model,
    ):
        if result.audio is not None:
            yield result.audio


async def run_tts(text: str, tts_config: KokoroConfig) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk_text in smart_split(text, tts_config.lang_code):
            if chunk_text.strip():
                async for audio in create_audio_stream(chunk_text, tts_config):
                    if audio is not None:
                        audio_data = audio * tts_config.volume_multiplier
                        yield (audio_data.cpu().numpy() * 32767).astype(
                            np.int16
                        ).tobytes()

                    else:
                        logger.warning("Empty audio chunk")
    except Exception as e:
        logger.info(f"TTS generation error: {e}")
        yield f"ERROR: {str(e)}".encode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting — loading model")
    await load_model()
    logger.info("Warming up model...")
    async for _ in run_tts("Hello, this is a test.", KokoroConfig()):
        break
    logger.info("Model pre-warmed.")
    yield
    logger.info("App shutting down — cleaning up")


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    try:
        while True:
            message = await websocket.receive_text()
            start_time = time.perf_counter()

            try:
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    config = KokoroConfig(**data.get("config", {}))
                except Exception:
                    text = message
                    config = KokoroConfig()

                if not text:
                    continue

                async for chunk in run_tts(text, config):
                    await websocket.send_bytes(chunk)

                await websocket.send_bytes("END")
                duration = time.perf_counter() - start_time
                logger.info(f"Processed in {duration:.2f}s")

            except Exception as e:
                logger.info(f"WebSocket error: {e}")
                await websocket.send_text(f"ERROR: {str(e)}")
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.info(f"WebSocket handler error: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8880, log_level="info", reload=False)
