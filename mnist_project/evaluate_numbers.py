"""Fixed synthetic benchmark: whole-number recognition, with oracle-crop control.

Only official test glyphs are composed. This is not a real multi-digit handwriting
dataset and is not used for tuning either segmentation or the frozen CNN.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from inference import load_checkpoint, predict_number, prepare_canvas, number_foreground
from models import normalize

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'/'numbers'


def compose_number(images,gap):
    canvas=np.zeros((120,500),dtype=np.uint8)
    x=20
    boxes=[]
    for image in images:
        ys,xs=np.where(image>40)
        crop=image[ys.min():ys.max()+1,xs.min():xs.max()+1]
        height=72
        width=max(4,round(crop.shape[1]*height/crop.shape[0]))
        glyph=np.asarray(Image.fromarray(crop).resize((width,height),Image.Resampling.BILINEAR))
        canvas[24:24+height,x:x+width]=np.maximum(canvas[24:24+height,x:x+width],glyph)
        boxes.append([x,24,width,height])
        x+=width+gap
    return canvas,boxes


def intersection_over_union(a,b):
    x0=max(a[0],b[0]); y0=max(a[1],b[1])
    x1=min(a[0]+a[2],b[0]+b[2]); y1=min(a[1]+a[3],b[1]+b[3])
    overlap=max(0,x1-x0)*max(0,y1-y0)
    return overlap/(a[2]*a[3]+b[2]*b[3]-overlap)


def run():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    selection=json.loads((ROOT/'artifacts'/'demo_selection.json').read_text())
    model,_=load_checkpoint(ROOT/'artifacts'/selection['checkpoint'])
    source=MNIST(ROOT/'data',train=False,download=True)
    labels=source.targets.numpy()
    pools={i:np.flatnonzero(labels==i) for i in range(10)}
    rng=np.random.default_rng(829)
    # Freeze glyph choices before observing any number scores; both layouts share them.
    choices={value:[int(rng.choice(pools[int(c)])) for c in str(value)] for value in range(1,1001)}
    (OUT/'protocol.json').write_text(json.dumps({'seed':829,'range':[1,1000],'glyph_height':72,
        'spaced_gap':18,'zero_gap':0,'iou_threshold':0.8,'source':'official MNIST test',
        'source_indices':choices,'checkpoint':selection['checkpoint']},indent=2),encoding='utf-8')
    rows=[]
    example_numbers=[7,42,407,1000]
    fig,axes=plt.subplots(2,4,figsize=(14,4.5))
    for layout,gap in [('spaced',18),('zero_gap',0)]:
        for value in range(1,1001):
            indices=choices[value]
            canvas,truth_boxes=compose_number([source.data[i].numpy() for i in indices],gap)
            try:
                result=predict_number(model,canvas)
                boxes=[d['bbox'] for d in result['digits']]
                predicted=result['text']
                valid=result['valid']
            except ValueError as exc:
                boxes=[]; predicted=''; valid=False
            good_segmentation=len(boxes)==len(truth_boxes) and all(intersection_over_union(a,b)>=.8 for a,b in zip(boxes,truth_boxes))
            # Oracle knows true digit boundaries, but uses the exact same crop preprocessing.
            oracle_images=[]
            clean_canvas,_=number_foreground(canvas)
            for x,y,w,h in truth_boxes:
                crop=clean_canvas[y:y+h,x:x+w]
                crop=np.pad(crop,((0,max(0,28-h)),(0,max(0,28-w))))
                oracle_images.append(prepare_canvas(crop))
            with torch.inference_mode():
                oracle=model(normalize(torch.cat(oracle_images))).argmax(1).tolist()
            oracle_text=''.join(map(str,oracle))
            rows.append({'layout':layout,'true_number':value,'predicted_text':predicted,'range_valid':valid,
                         'exact_number':predicted==str(value) and valid,'true_digits':len(str(value)),
                         'found_digits':len(boxes),'segmentation_correct':good_segmentation,
                         'oracle_text':oracle_text,'oracle_exact':oracle_text==str(value),
                         'oracle_digits_correct':sum(a==b for a,b in zip(oracle_text,str(value)))})
            if value in example_numbers:
                ax=axes[0 if layout=='spaced' else 1,example_numbers.index(value)]
                ax.imshow(canvas,cmap='gray',vmin=0,vmax=255)
                for x,y,w,h in boxes:
                    ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,edgecolor='#70d6ba',linewidth=1))
                ax.set_title(f'{value} → {predicted or "помилка"} · {layout}')
                ax.set_xlim(5,min(500,max(b[0]+b[2] for b in truth_boxes)+20)); ax.axis('off')
        print(f'Finished {layout}: all integers 1–1000',flush=True)
    fig.suptitle('Синтетичні числа з test-цифр MNIST: приклади обох умов')
    plt.tight_layout(); plt.savefig(OUT/'examples.png',dpi=150,bbox_inches='tight'); plt.close()
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'predictions.csv',index=False)
    summary={'source':'official_mnist_test_digits_synthetic_composition','checkpoint':selection['checkpoint'],
             'real_handwriting_evaluated':False,'layouts':{}}
    for layout in ['spaced','zero_gap']:
        f=frame[frame.layout==layout]
        summary['layouts'][layout]={'samples':len(f),'numbers_min':int(f.true_number.min()),'numbers_max':int(f.true_number.max()),
            'exact_numbers':int(f.exact_number.sum()),'exact_number_accuracy':float(f.exact_number.mean()),
            'segmentation_accuracy_iou08':float(f.segmentation_correct.mean()),'digit_count_accuracy':float((f.found_digits==f.true_digits).mean()),
            'oracle_exact_number_accuracy':float(f.oracle_exact.mean()),'oracle_digits_correct':int(f.oracle_digits_correct.sum()),
            'oracle_digits_total':int(f.true_digits.sum()),'oracle_digit_accuracy':float(f.oracle_digits_correct.sum()/f.true_digits.sum()),
            'range_rejections':int((~f.range_valid).sum())}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    run()
