"""Train fixed experiments. Test scores never enter checkpoint/model selection."""
import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import load_training, augment_batch, raw_hashes
from models import make_model, normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate_validation(model, x, y):
    model.eval()
    losses, predictions = 0., []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            logits = model(normalize(x[start:start+512]))
            losses += nn.functional.cross_entropy(logits, y[start:start+512], reduction='sum').item()
            predictions.extend(logits.argmax(1).tolist())
    return losses / len(x), accuracy_score(y, predictions), f1_score(y, predictions, average='macro')


def run(config_path=ROOT / 'config.json'):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=True)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != cfg:
        raise ValueError('Existing artifacts use a different protocol. Choose a fresh output directory.')
    protocol_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    torch.set_num_threads(cfg['threads'])
    x, y, xv, yv, a, b = load_training(validation_size=cfg['validation_size'], split_seed=cfg['split_seed'])
    np.savez_compressed(OUT / 'split.npz', train=a, validation=b)
    versions = {'python': platform.python_version(), 'torch': torch.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'device': 'cpu', 'threads': cfg['threads']}
    (OUT / 'environment.json').write_text(json.dumps({'versions': versions, 'raw_sha256': raw_hashes()}, indent=2), encoding='utf-8')
    records = []
    for name, architecture, augmentation in [('MLP', 'mlp', False), ('CNN', 'cnn', False), ('CNN_aug', 'cnn', True)]:
        for seed in cfg['seeds']:
            key = f'{name}_{seed}'
            metadata_path = OUT / f'{key}.json'
            if metadata_path.exists():
                records.append(json.loads(metadata_path.read_text()))
                print(f'Reuse completed {key}', flush=True)
                continue
            set_seed(seed)
            model = make_model(architecture)
            initial_sha = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            loader = DataLoader(TensorDataset(x, y), batch_size=cfg['batch_size'], shuffle=True,
                                generator=torch.Generator().manual_seed(seed), num_workers=0)
            aug_rng = torch.Generator().manual_seed(seed + 10000)
            best, history, selected_epoch = -1., [], 0
            start = time.perf_counter()
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                total_loss, correct, total = 0., 0, 0
                for xb, yb in loader:
                    if augmentation:
                        aug = cfg['augmentation']
                        xb = augment_batch(xb, aug_rng, aug['rotation_degrees'], aug['translation_fraction'], aug['scale'])
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(normalize(xb))
                    loss = nn.functional.cross_entropy(logits, yb)
                    loss.backward()  # PyTorch autograd; no handwritten backpropagation
                    optimizer.step()
                    total_loss += loss.item() * len(yb)
                    correct += int((logits.argmax(1) == yb).sum())
                    total += len(yb)
                val_loss, val_accuracy, val_f1 = evaluate_validation(model, xv, yv)
                row = {'epoch': epoch, 'train_loss': total_loss/total, 'train_accuracy': correct/total,
                       'validation_loss': val_loss, 'validation_accuracy': val_accuracy, 'validation_macro_f1': val_f1}
                history.append(row)
                if val_f1 > best:
                    best, selected_epoch = val_f1, epoch
                    checkpoint = {'state_dict': copy.deepcopy(model.state_dict()), 'architecture': architecture,
                                  'name': name, 'seed': seed, 'epoch': epoch, 'validation_macro_f1': float(best)}
                    torch.save(checkpoint, OUT / 'checkpoints' / f'{key}.pt')
                print(f'{key} epoch={epoch}/{cfg["epochs"]} loss={row["train_loss"]:.4f} val_f1={val_f1:.4f}', flush=True)
            pd.DataFrame(history).to_csv(OUT / f'history_{key}.csv', index=False)
            metadata = {'name': name, 'seed': seed, 'validation_macro_f1': float(best), 'selected_epoch': selected_epoch,
                        'training_seconds': time.perf_counter()-start, 'parameters': sum(p.numel() for p in model.parameters()),
                        'checkpoint': f'checkpoints/{key}.pt', 'epochs_run': len(history), 'initial_weights_sha256': initial_sha}
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            records.append(metadata)
    baseline_path = OUT / 'baseline.json'
    if baseline_path.exists():
        baseline_meta = json.loads(baseline_path.read_text())
    else:
        start = time.perf_counter()
        baseline = LogisticRegression(C=1., max_iter=300, solver='lbfgs', tol=1e-3)
        baseline.fit(x.flatten(1).numpy(), y.numpy())
        prediction = baseline.predict(xv.flatten(1).numpy())
        baseline_meta = {'name': 'LogisticRegression', 'seed': 42, 'validation_macro_f1': f1_score(yv, prediction, average='macro'),
                         'training_seconds': time.perf_counter()-start, 'parameters': int(baseline.coef_.size + baseline.intercept_.size)}
        joblib.dump(baseline, OUT / 'logistic_regression.joblib')
        baseline_path.write_text(json.dumps(baseline_meta, indent=2), encoding='utf-8')
    # The demonstration checkpoint is picked before loading the official test set.
    selected = max(records, key=lambda r: r['validation_macro_f1'])
    (OUT / 'demo_selection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    pd.DataFrame(records + [baseline_meta]).to_csv(OUT / 'training_summary.csv', index=False)
    print('Training complete; demo selected by validation:', selected['checkpoint'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    run(parser.parse_args().config)
