#!/usr/bin/env python3

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    repo_type="model",
    local_dir="./kokoro_model",
    allow_patterns=[ "kokoro-v1_0.pth","config.json","voices/*",]
)
