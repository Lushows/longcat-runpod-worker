# LongCat-Video-Avatar 1.5 worker para RunPod serverless.
# OJO GPU: necesita ~48GB VRAM (A6000/A40 48GB). torch 2.6 cu124 -> Ampere/Ada, NO Blackwell.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git git-lfs wget ffmpeg build-essential cmake ninja-build libgl1 libglib2.0-0 libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && git lfs install

# Código de LongCat-Video (incluye los demos de avatar)
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /app/longcat
WORKDIR /app/longcat

# PyTorch 2.6 (cu124, Ampere/Ada) — pasos SEPARADOS para aislar errores
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir psutil packaging ninja
# flash_attn PRE-COMPILADO (wheel listo: cp310 + torch2.6 + cu12) -> NO necesita nvcc/compilar
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_avatar.txt || true
RUN pip install --no-cache-dir librosa soundfile boto3 runpod requests "huggingface_hub[cli]"

# Pesos del modelo (~30GB) desde HuggingFace
RUN huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir ./weights/LongCat-Video-Avatar-1.5

# Nuestro handler + util de subida a R2
COPY app/ /app/longcat/app/

CMD ["python", "-u", "/app/longcat/app/handler.py"]
