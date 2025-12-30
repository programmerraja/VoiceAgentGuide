FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN pip install faster-whisper websockets numpy torch uvicorn fastapi

COPY download_model.py .
# RUN pip install huggingface_hub
RUN python3 download_model.py

WORKDIR /app
COPY server.py .

EXPOSE 8000
RUN export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib


CMD ["python3", "server.py"]
