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
# requirements_avatar.txt trae 2 lineas TOXICAS que rompen TODO el install (resolver de pip
# falla entero -> no instala NADA -> faltaban audio-separator, pyloudnorm, onnx, etc. una por una):
#   - libsndfile1==0.0.1     -> NO existe en PyPI (es lib de sistema, ya viene por apt)
#   - tritonserverclient==0.0.6 -> NO existe en PyPI (404)
# Las quitamos junto con torch/flash (que pinean 2.6) y dejamos instalar el resto COMPLETO.
# SIN '|| true': si algo falla, que reviente el BUILD (no en runtime perdiendo tiempo).
# Tambien quitamos onnxruntime: LongCat pinea ==1.16.3 (2023) cuyo .so tiene marcada bandera
# de "executable stack" -> ImportError "cannot enable executable stack" al importarlo. onnxruntime
# >=1.17.0 ya NO trae esa bandera (fix oficial release notes 1.17). Instalamos 1.19.2 abajo.
RUN if [ -f requirements_avatar.txt ]; then sed -i -E '/^(torch|torchvision|torchaudio|flash[-_]attn|libsndfile1|tritonserverclient|onnxruntime)([=<>! ]|$)/d' requirements_avatar.txt; fi
RUN pip install --no-cache-dir -r requirements_avatar.txt
# onnxruntime nuevo (sin executable stack); audio-separator de hecho prefiere >=1.17
RUN pip install --no-cache-dir onnxruntime==1.19.2
# Deps propias del worker + 3 que el codigo de avatar importa pero NINGUN requirements declara
# (auditoria del grafo de imports 2026-06-05): regex (pipeline), tqdm (pipeline/audio_process),
# triton (block_sparse_attention; suele venir con torch 2.7 pero lo verificamos abajo).
RUN pip install --no-cache-dir boto3 runpod requests regex tqdm "huggingface_hub[hf_transfer]"

# triton lo trae torch 2.7 (pytorch-triton). Si por lo que sea no esta, instalarlo sin romper torch.
RUN python -c "import triton" || pip install --no-cache-dir triton

# VERIFICACION EN BUILD: importa TODO el set de deps del camino de avatar (ai2v + use_int8 +
# avatar-v1.5 + use_distill). Si falta UNA, el build REVIENTA aqui (visible en Actions) en vez de
# fallar en runtime tras 8 min de cold start. Asi, build verde == todas las deps presentes.
RUN python -c "import triton, regex, tqdm, audio_separator, pyloudnorm, librosa, soundfile, soxr, scipy, sklearn, skimage, transformers, diffusers, einops, loguru, ftfy, imageio, imageio_ffmpeg, onnx, onnxruntime, numpy, PIL, torchvision, flash_attn; print('=== SMOKE IMPORT OK: todas las deps de avatar presentes ===')"

# NOTA: el modelo (~30GB) NO se hornea en la imagen (reventaba el build por tamaño/limite de 30min
# y daba "input/output error" al escribir la capa gigante). Se descarga en runtime a un
# Network Volume montado en /runpod-volume (una sola vez, lo reusan todos los workers).

# Nuestro handler + util de subida a R2
COPY app/ /app/longcat/app/

CMD ["python", "-u", "/app/longcat/app/handler.py"]
# trigger build 20260605133800-blackwell-force
