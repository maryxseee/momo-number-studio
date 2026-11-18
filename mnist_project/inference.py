"""Shared inference for CLI tests and the local drawing application."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass, shift, label
from models import make_model, normalize


def prepare_canvas(pixels):
    try:
        a = np.asarray(pixels, dtype=np.float32)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Некоректний масив пікселів.') from exc
    if a.ndim != 2 or not (28 <= a.shape[0] <= 560 and 28 <= a.shape[1] <= 560):
        raise ValueError('Потрібне квадратне або прямокутне полотно 28–560 пікселів.')
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError('Некоректні значення пікселів.')
    mask = a > 30
    if mask.sum() < 10:
        raise ValueError('Спочатку намалюйте одну цифру.')
    yy, xx = np.where(mask)
    crop = a[yy.min():yy.max()+1, xx.min():xx.max()+1]
    h, w = crop.shape
    scale = 20 / max(h, w)
    resized = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    result = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28-resized.shape[0])//2, (28-resized.shape[1])//2
    result[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized
    cy, cx = center_of_mass(result)
    result = shift(result, (13.5-cy, 13.5-cx), order=1, mode='constant', cval=0).clip(0, 1)
    return torch.from_numpy(result.copy()).reshape(1, 1, 28, 28)


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = make_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def predict_canvas(model, pixels):
    image = prepare_canvas(pixels)
    with torch.inference_mode():
        probabilities = model(normalize(image)).softmax(dim=1)[0]
    top = torch.topk(probabilities, 3)
    return {'top3': [{'digit': int(i), 'score': float(p)} for p, i in zip(top.values, top.indices)],
            'preview': (image[0, 0].numpy() * 255).astype(int).tolist()}


def number_foreground(pixels):
    """Shared validation and foreground filtering for inference and oracle control."""
    try:
        a=np.asarray(pixels,dtype=np.float32)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('Некоректний масив пікселів.') from exc
    if a.ndim!=2 or not (28<=a.shape[0]<=560 and 28<=a.shape[1]<=1120):
        raise ValueError('Потрібне полотно 28–560 пікселів заввишки та 28–1120 завширшки.')
    if not np.isfinite(a).all() or a.min()<0 or a.max()>255:
        raise ValueError('Некоректні значення пікселів.')
    mask=a>40
    components,count=label(mask,structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel())
    keep=sizes>=max(6,int(mask.sum()*.002))
    keep[0]=False
    mask=keep[components]
    if mask.sum()<10:
        raise ValueError('Спочатку намалюйте число від 1 до 1000.')
    return np.where(mask,a,0),mask


def segment_number(pixels):
    """Segment a spaced row using blank columns, not digit classification.

    Tiny 8-connected components are removed. Internal gaps up to max(4px,3.5%
    of ink height) are joined. Touching/adjacent glyphs cannot be reliably split.
    Boxes use [x,y,width,height].
    """
    clean,mask=number_foreground(pixels)
    ys=np.flatnonzero(mask.any(axis=1))
    merge_gap=max(4,round((ys[-1]-ys[0]+1)*.035))
    occupied=np.flatnonzero(mask.any(axis=0))
    groups=np.split(occupied,np.flatnonzero(np.diff(occupied)>merge_gap+1)+1)
    if len(groups)>4:
        raise ValueError('Знайдено більше чотирьох цифр. Напишіть одне число 1–1000.')
    parts=[]
    for columns in groups:
        x0,x1=int(columns[0]),int(columns[-1])+1
        rows=np.flatnonzero(mask[:,x0:x1].any(axis=1))
        y0,y1=int(rows[0]),int(rows[-1])+1
        crop=clean[y0:y1,x0:x1]
        # The existing preparation accepts arrays >=28 px on each axis.
        crop=np.pad(crop,((0,max(0,28-crop.shape[0])),(0,max(0,28-crop.shape[1]))))
        parts.append({'bbox':[x0,y0,x1-x0,y1-y0],'crop':crop})
    return parts


def validate_number(text):
    if not text or not text.isascii() or not text.isdigit() or len(text)>4:
        return None,'Не вдалося отримати число з однієї–чотирьох цифр.'
    if len(text)>1 and text[0]=='0':
        return None,'Провідні нулі не підтримуються. Напишіть число без них.'
    value=int(text)
    if not 1<=value<=1000:
        return None,'Розпізнане число поза діапазоном 1–1000.'
    return value,None


def predict_number(model,pixels):
    parts=segment_number(pixels)
    images=torch.cat([prepare_canvas(p['crop']) for p in parts])
    with torch.inference_mode():
        probabilities=model(normalize(images)).softmax(1)
    digits=[]
    for part,image,probability in zip(parts,images,probabilities):
        top=torch.topk(probability,3)
        digits.append({'bbox':part['bbox'],
                       'top3':[{'digit':int(i),'score':float(p)} for p,i in zip(top.values,top.indices)],
                       'preview':(image[0].numpy()*255).astype(int).tolist()})
    text=''.join(str(d['top3'][0]['digit']) for d in digits)
    number,error=validate_number(text)
    return {'text':text,'number':number,'valid':error is None,'message':error,
            'segmentation_count':len(digits),'digits':digits}
