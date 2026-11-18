"""Vendor ONNX Runtime Web and export the existing validation-selected model."""
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request
import hashlib

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'


def vendor_runtime():
    metadata=json.load(urllib.request.urlopen('https://registry.npmjs.org/onnxruntime-web/1.31.1',timeout=30))
    tar_url=metadata['dist']['tarball']
    if not tar_url.startswith('https://registry.npmjs.org/onnxruntime-web/'):
        raise ValueError('Unexpected runtime registry host')
    raw=urllib.request.urlopen(tar_url,timeout=60).read()
    destination=SITE/'vendor'
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        files=['dist/ort.wasm.min.js','dist/ort-wasm-simd-threaded.mjs','dist/ort-wasm-simd-threaded.wasm']
        for name in files:
            content=archive.extractfile('package/'+name).read()
            (destination/Path(name).name).write_bytes(content)
    # Keep the reviewed upstream MIT notice; npm omits it from its tarball.
    license_path=destination/'LICENSE'
    if not license_path.exists():
        raise RuntimeError('Restore site/vendor/LICENSE from this repository before rebuilding runtime assets')
    (destination/'version.json').write_text(json.dumps({'package':'onnxruntime-web','version':metadata['version'],'source':tar_url},indent=2))
    print('Vendored ONNX Runtime Web',metadata['version'],flush=True)


def export_model():
    import numpy as np
    import torch
    import onnx
    sys.path.insert(0,str(ROOT/'mnist_project'))
    from inference import load_checkpoint
    selection=json.loads((ROOT/'mnist_project/artifacts/demo_selection.json').read_text())
    model,metadata=load_checkpoint(ROOT/'mnist_project/artifacts'/selection['checkpoint'])
    class WebModel(torch.nn.Module):
        def __init__(self,model):
            super().__init__(); self.features=model[:-1]; self.classifier=model[-1]
        def forward(self,pixels):
            features=self.features((pixels-.5)/.5)
            return self.classifier(features).softmax(1),features
    wrapper=WebModel(model).eval()
    assets=SITE/'assets'; assets.mkdir(parents=True,exist_ok=True)
    path=assets/'digit-cnn.onnx'
    torch.onnx.export(wrapper,torch.zeros(1,1,28,28),path,input_names=['pixels'],output_names=['probabilities','features'],
                      dynamic_axes={'pixels':{0:'batch'},'probabilities':{0:'batch'},'features':{0:'batch'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    rng=np.random.default_rng(829)
    inputs=rng.random((3,1,28,28),dtype=np.float32)
    with torch.inference_mode():
        probabilities,features=wrapper(torch.from_numpy(inputs))
        expected=probabilities.numpy()
    (assets/'parity.json').write_text(json.dumps({'shape':list(inputs.shape),'inputs':inputs.ravel().tolist(),'expected':expected.tolist()}))
    evaluation=json.loads((ROOT/'mnist_project/artifacts/evaluation.json').read_text())
    chosen=next(r for r in evaluation['runs'] if r['name']==metadata['name'] and r['seed']==metadata['seed'])
    cnn_runs=[r['conditions']['clean']['accuracy'] for r in evaluation['runs'] if r['name']=='CNN']
    number_results=json.loads((ROOT/'mnist_project/artifacts/numbers/summary.json').read_text())
    public={'name':metadata['name'],'seed':metadata['seed'],'epoch':metadata['epoch'],
            'checkpoint_selection':'validation_macro_f1','single_digit_test_accuracy':chosen['conditions']['clean']['accuracy'],
            'research_mean_digit_accuracy':sum(cnn_runs)/len(cnn_runs),'python_synthetic_number_accuracy':number_results['layouts']['spaced']['exact_number_accuracy'],
            'browser_number_accuracy':None,'browser_preprocessing':'canvas resampling + center of mass; differs from PIL/SciPy',
            'runtime':'ONNX Runtime Web','input':'float32 [N,1,28,28], range 0..1; normalization is inside graph'}
    (assets/'model.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    from torchvision.datasets import MNIST
    from evaluate_numbers import compose_number
    from PIL import Image
    choices=json.loads((ROOT/'mnist_project/artifacts/numbers/protocol.json').read_text())['source_indices']['407']
    source=MNIST(ROOT/'mnist_project/data',train=False,download=False)
    canvas,_=compose_number([source.data[i].numpy() for i in choices],18)
    rgba=np.zeros((*canvas.shape,4),dtype=np.uint8)
    rgba[:,:,:3]=[64,40,65];rgba[:,:,3]=canvas
    Image.fromarray(rgba).save(assets/'example-407.png')
    tag=hashlib.sha256(path.read_bytes()).hexdigest()
    head={'modelTag':tag,'featureSize':64,'kernel':model[-1].weight.detach().numpy().T.ravel().tolist(),'bias':model[-1].bias.detach().numpy().tolist()}
    (assets/'head-initial.json').write_text(json.dumps(head),encoding='utf-8')
    training_source=MNIST(ROOT/'mnist_project/data',train=True,download=False)
    split=np.load(ROOT/'mnist_project/artifacts/split.npz')['train']
    labels=training_source.targets.numpy();rng=np.random.default_rng(829)
    replay_ids=[];guard_ids=[]
    for digit in range(10):
        selected=rng.choice(split[labels[split]==digit],40,replace=False)
        replay_ids.extend(selected[:30].tolist());guard_ids.extend(selected[30:].tolist())
    torch.set_num_threads(2)
    replay={'modelTag':tag,'source':'MNIST original training subset only; excludes validation/test','seed':829}
    for name,ids in [('train',replay_ids),('guard',guard_ids)]:
        images=training_source.data[ids].unsqueeze(1).float()/255
        with torch.inference_mode():
            _,encoded=wrapper(images)
        replay[name]={'features':encoded.numpy().tolist(),'labels':labels[ids].tolist(),'sourceIndices':ids}
    (assets/'head-replay.json').write_text(json.dumps(replay),encoding='utf-8')
    print('Exported checked model:',path.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    vendor_runtime()
    export_model()
