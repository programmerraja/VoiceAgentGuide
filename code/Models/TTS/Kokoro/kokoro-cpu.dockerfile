FROM python:3.10-slim

ENV HF_HOME=/root/.cache/huggingface/hub

RUN apt-get update 

RUN pip install numpy websockets asyncio loguru kokoro inflect pydantic torch 
RUN pip install fastapi uvicorn

COPY . .

EXPOSE 9802

RUN python3 download_model.py

CMD ["python3", "server.py"]