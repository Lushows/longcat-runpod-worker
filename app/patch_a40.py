# PATCH A40: el DiT de 13.6B se construye en bf16 (~27GB RAM) en vez de fp32 (~54GB) -> cabe en la
# RAM de una A40 48GB (~50GB), que es 2.2x mas barata que A100/H100. Sin esto, OOM de RAM al instanciar.
import sys
p = '/app/longcat/longcat_video/modules/quantization.py'
s = open(p).read()
old = '    model = LongCatVideoAvatarTransformer3DModel(**config)'
new = ('    _pd = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)  # PATCH A40: bf16 (27GB RAM) vs fp32 (54GB)\n'
       '    model = LongCatVideoAvatarTransformer3DModel(**config)\n'
       '    torch.set_default_dtype(_pd)')
if old not in s:
    sys.exit('PATCH A40: no encontre la linea de construccion del DiT en ' + p)
open(p, 'w').write(s.replace(old, new, 1))
print('=== PATCH A40 OK: el DiT se construye en bf16 (cabe en A40 48GB) ===')
