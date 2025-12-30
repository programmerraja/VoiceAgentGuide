FROM python:3.11-slim

RUN apt-get update 

RUN pip install faster-whisper websockets numpy uvicorn fastapi
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY download_model.py .
# RUN pip install huggingface_hub
RUN python3 download_model.py


WORKDIR /app
COPY server.py .

EXPOSE 9800 

CMD ["python3", "server.py"]
