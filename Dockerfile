# LongCat-Video-Avatar 1.5 worker para RunPod serverless.
# GPU: BLACKWELL (RTX 5090 32GB / RTX PRO 6000 96GB). torch 2.7 cu128 -> soporta sm_120.
# El modelo INT8 cabe en 32GB. Se eligio Blackwell por disponibilidad+precio en RunPod 2026.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git git-lfs wget ffmpeg build-essential cmake ninja-build libgl1 libglib2.0-0 libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && git lfs install

# Código de LongCat-Video (incluye los demos de avatar)
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /app/longcat
WORKDIR /app/longcat

# PyTorch 2.7 (cu128, Blackwell sm_120) — pasos SEPARADOS para aislar errores
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir psutil packaging ninja
# flash_attn PRE-COMPILADO (wheel listo: cp310 + torch2.7 + cu12) -> NO necesita nvcc/compilar
# Si diera "undefined symbol" en runtime, cambiar abiFALSE -> abiTRUE en esta linea.
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# Quitar torch/torchvision/torchaudio/flash-attn de requirements para que NO degraden a 2.6
RUN sed -i -E '/^(torch|torchvision|torchaudio|flash[-_]attn)([=<>! ]|$)/d' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ -f requirements_avatar.txt ]; then sed -i -E '/^(torch|torchvision|torchaudio|flash[-_]attn)([=<>! ]|$)/d' requirements_avatar.txt; fi
RUN pip install --no-cache-dir -r requirements_avatar.txt || true
RUN pip install --no-cache-dir librosa soundfile pyloudnorm boto3 runpod requests "huggingface_hub[hf_transfer]"

# NOTA: el modelo (~30GB) NO se hornea en la imagen (reventaba el build por tamaño/limite de 30min
# y daba "input/output error" al escribir la capa gigante). Se descarga en runtime a un
# Network Volume montado en /runpod-volume (una sola vez, lo reusan todos los workers).

# Nuestro handler + util de subida a R2
COPY app/ /app/longcat/app/

CMD ["python", "-u", "/app/longcat/app/handler.py"]
# trigger build 20260605133800-blackwell-force
