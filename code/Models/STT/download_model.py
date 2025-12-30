#!/usr/bin/env python3

from huggingface_hub import snapshot_download

snapshot_download(
    # repo_id="Systran/faster-whisper-tiny",
    repo_id="Systran/faster-distil-whisper-medium.en",
    repo_type="model",
    local_dir="./app/whisper_model",
)
