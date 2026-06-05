import os
import sys
import json
import time
import glob
import subprocess

import runpod

sys.path.insert(0, os.path.dirname(__file__))
from file_utils import download_file, upload_to_s3

REPO = '/app/longcat'
CKPT = '/app/longcat/weights/LongCat-Video-Avatar-1.5'
INDIR = '/app/longcat/_in'

DEFAULT_PROMPT = ('A news anchor talks to the camera with natural hand gestures and '
                  'lively facial expressions, professional studio lighting.')

def newest_mp4(since_ts):
    cands = []
    for p in glob.glob(REPO + '/**/*.mp4', recursive=True):
        try:
            if os.path.getmtime(p) >= since_ts - 1:
                cands.append(p)
        except Exception:
            pass
    if not cands:
        cands = glob.glob(REPO + '/**/*.mp4', recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None

def handler(job):
    ji = job.get('input', {})
    img_url = ji.get('input_image_url')
    aud_url = ji.get('input_audio_url')
    prompt = (ji.get('prompt') or DEFAULT_PROMPT)[:125]
    if not img_url:
        return {'error': 'falta input_image_url'}
    if not aud_url:
        return {'error': 'falta input_audio_url'}

    os.makedirs(INDIR, exist_ok=True)
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
        '--context_parallel_size=1', f'--checkpoint_dir={CKPT}', '--stage_1=ai2v',
        f'--input_json={jpath}', '--use_distill', '--model_type', 'avatar-v1.5', '--use_int8',
    ]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or '')[-900:]
        return {'error': 'LongCat falló: ' + tail}

    out = newest_mp4(t0)
    if not out:
        return {'error': 'no se encontró el video de salida. ' + (p.stdout or '')[-400:]}

    obj = 'longcat-' + str(int(t0)) + '.mp4'
    url, e = upload_to_s3(out, os.getenv('BUCKET_NAME', 'studio-lipsync'), obj)
    if e:
        return {'error': f'subida a R2 falló: {e}'}
    return {'output_video_url': url}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
