import os
import sys
import json
import time
import glob
import shutil
import traceback
import subprocess

import runpod

sys.path.insert(0, os.path.dirname(__file__))
from file_utils import download_file, upload_to_s3

os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

REPO = '/app/longcat'
INDIR = '/app/longcat/_in'
OUTDIR = '/app/longcat/outputs_avatar_single'

# El modelo vive en el Network Volume (/runpod-volume) si está montado; si no, en disco local.
VOL = '/runpod-volume' if os.path.isdir('/runpod-volume') else '/app/longcat/weights'
CKPT = os.path.join(VOL, 'LongCat-Video-Avatar-1.5')
_DONE = os.path.join(CKPT, '.download_complete')

DEFAULT_PROMPT = ('A news anchor talks to the camera with natural hand gestures and '
                  'lively facial expressions, professional studio lighting.')


def log(*a):
    print('[longcat]', *a, flush=True)


def disk_free(path):
    try:
        p = path if os.path.isdir(path) else os.path.dirname(path) or '/'
        t, u, f = shutil.disk_usage(p)
        return f'{f / 1e9:.1f}GB libres de {t / 1e9:.1f}GB en {p}'
    except Exception as e:
        return f'? ({e})'


def ensure_model():
    """Descarga el modelo (~30GB) UNA sola vez al volumen. Los siguientes workers lo reusan."""
    log('volume_mounted(/runpod-volume)=', os.path.isdir('/runpod-volume'), '| VOL=', VOL, '| CKPT=', CKPT)
    log('disco destino:', disk_free(VOL))
    if os.path.exists(_DONE):
        log('modelo ya presente en el volumen -> sin descarga')
        return CKPT, None
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(CKPT, exist_ok=True)
        log('descargando SOLO los pesos int8 (~20GB) a', CKPT, '...')
        # Saltar lo que NO se usa con --use_int8 (ahorra ~30GB): base FP32, formatos whisper
        # redundantes, videos demo. Asi cabe en disco efimero y baja mucho mas rapido.
        snapshot_download(
            repo_id='meituan-longcat/LongCat-Video-Avatar-1.5',
            local_dir=CKPT,
            max_workers=8,
            ignore_patterns=[
                'diffusion_pytorch_model-*',  # base FP32 (29.5GB) -> usamos quantized para int8
                '*fp32*',                     # formatos whisper fp32 redundantes
                'flax_model.msgpack',         # whisper flax (5.75GB)
                '*.mp4',                      # videos demo (9.3GB)
                'assets/*',                   # logos/demos del repo
            ],
        )
        open(_DONE, 'w').close()
        log('modelo descargado OK ->', disk_free(VOL))
        return CKPT, None
    except Exception as e:
        log('ERROR descargando modelo:', repr(e))
        return None, str(e)


def patch_attention_config(ckpt):
    """LongCat trae config.json con enable_flashattn2=true, pero flash_attn 2.7.4.post1 esta roto
    con torch 2.7 (ABI). Forzamos enable_xformers=true (xformers SI es compatible con torch 2.7 y
    el codigo tiene branch xformers para los 3 sitios de atencion). Idempotente: se corre siempre."""
    for sub in ('base_model_int8', 'base_model'):
        cfgpath = os.path.join(ckpt, sub, 'config.json')
        if not os.path.exists(cfgpath):
            continue
        try:
            with open(cfgpath) as f:
                cfg = json.load(f)
            cfg['enable_flashattn3'] = False
            cfg['enable_flashattn2'] = False
            cfg['enable_xformers'] = True
            cfg['enable_bsa'] = False
            with open(cfgpath, 'w') as f:
                json.dump(cfg, f, indent=2)
            log('config de atencion -> xformers:', cfgpath)
        except Exception as e:
            log('WARN no pude patchear', cfgpath, ':', repr(e))


def newest_mp4(since_ts):
    cands = []
    for root in (OUTDIR, REPO):
        for p in glob.glob(root + '/**/*.mp4', recursive=True):
            try:
                if os.path.getmtime(p) >= since_ts - 1:
                    cands.append(p)
            except Exception:
                pass
        if cands:
            break
    if not cands:
        cands = glob.glob(REPO + '/**/*.mp4', recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None


def handler(job):
    try:
        ji = job.get('input', {})
        img_url = ji.get('input_image_url')
        aud_url = ji.get('input_audio_url')
        prompt = (ji.get('prompt') or DEFAULT_PROMPT)[:125]
        log('=== nuevo job === img?', bool(img_url), 'aud?', bool(aud_url))
        if not img_url:
            return {'error': 'falta input_image_url'}
        if not aud_url:
            return {'error': 'falta input_audio_url'}

        ckpt, e = ensure_model()
        if e:
            return {'error': f'descarga del modelo fallo: {e}'}
        patch_attention_config(ckpt)  # flash_attn roto en torch 2.7 -> forzar xformers

        os.makedirs(INDIR, exist_ok=True)
        os.makedirs(OUTDIR, exist_ok=True)
        log('descargando imagen y audio de entrada...')
        img, e = download_file(img_url, os.path.join(INDIR, 'img.png'))
        if e:
            return {'error': f'no pude descargar la imagen: {e}'}
        aud, e = download_file(aud_url, os.path.join(INDIR, 'aud.wav'))
        if e:
            return {'error': f'no pude descargar el audio: {e}'}

        cfg = {'prompt': prompt, 'cond_image': img, 'cond_audio': {'person1': aud}}
        jpath = os.path.join(INDIR, 'job.json')
        with open(jpath, 'w') as f:
            json.dump(cfg, f)

        t0 = time.time()
        cmd = [
            'torchrun', '--nproc_per_node=1', 'run_demo_avatar_single_audio_to_video.py',
            '--context_parallel_size=1', f'--checkpoint_dir={ckpt}', '--stage_1=ai2v',
            f'--input_json={jpath}', f'--output_dir={OUTDIR}',
            '--use_distill', '--model_type', 'avatar-v1.5', '--use_int8',
        ]
        log('lanzando generacion:', ' '.join(cmd))
        # SIN capture_output -> la salida del torchrun se transmite EN VIVO a los logs de RunPod.
        p = subprocess.run(cmd, cwd=REPO)
        log('torchrun termino con returncode =', p.returncode)
        if p.returncode != 0:
            return {'error': f'LongCat fallo (returncode {p.returncode}). Revisa los logs de RunPod para el detalle.'}

        out = newest_mp4(t0)
        if not out:
            return {'error': 'no se encontro el video de salida (.mp4) tras la generacion'}
        log('video generado:', out)

        obj = 'longcat-' + str(int(t0)) + '.mp4'
        url, e = upload_to_s3(out, os.getenv('BUCKET_NAME', 'studio-lipsync'), obj)
        if e:
            return {'error': f'subida a R2 fallo: {e}'}
        log('subido a R2:', url)
        return {'output_video_url': url, 'seconds': round(time.time() - t0, 1)}
    except Exception as ex:
        tb = traceback.format_exc()
        log('EXCEPCION no controlada en handler:\n', tb)
        return {'error': 'excepcion en el worker: ' + str(ex)}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
