FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
    
RUN pip install numpy \
    asyncio \
    loguru \
    kokoro  \
    inflect \
    pydantic \
    fastapi \
    uvicorn  

RUN pip install websockets


COPY . .

EXPOSE 9802

RUN python3 download_model.py

CMD ["python3", "server.py"]