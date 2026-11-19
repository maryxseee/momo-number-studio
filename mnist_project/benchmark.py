"""Repeated CPU inference benchmark of frozen checkpoints; never trains/tunes models."""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from data import load_test
from inference import load_checkpoint
from models import normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
REPEATS = 7
WARMUPS = 3


def measure(predict, images, batch_size, expected):
    """Return raw wall times; correctness checked outside the timed region."""
    predicted = np.concatenate([predict(images[i:i+batch_size])
                                for i in range(0, len(images), batch_size)])
    np.testing.assert_array_equal(predicted, expected)
    for _ in range(WARMUPS):
        predict(images[:batch_size])
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for i in range(0, len(images), batch_size):
            predict(images[i:i+batch_size])
        timings.append(time.perf_counter() - start)
    return timings


def run():
    cfg = json.loads((OUT / 'protocol.json').read_text())
    torch.set_num_threads(cfg['threads'])
    training = pd.read_csv(OUT / 'training_summary.csv')
    images, _ = load_test()
    saved = np.load(OUT / 'predictions.npz')
    records, files = [], {}
    with threadpool_limits(limits=cfg['threads']):
        pools = threadpool_info()
        for row in training.to_dict('records'):
            name, seed = row['name'], int(row['seed'])
            path = OUT / ('logistic_regression.joblib' if name == 'LogisticRegression' else row['checkpoint'])
            files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == 'LogisticRegression':
                model = joblib.load(path)
                def predict(batch):
                    return model.predict_proba(batch.flatten(1).numpy()).argmax(1)
            else:
                model, _ = load_checkpoint(path)
                def predict(batch):
                    with torch.inference_mode():
                        return model(normalize(batch)).softmax(1).argmax(1).numpy()
            expected = saved[f'{name}_{seed}__clean']
            for mode, batch_size, count in [('single', 1, 100), ('batch', 256, 10000)]:
                times = measure(predict, images[:count], batch_size, expected[:count])
                for repeat, seconds in enumerate(times, 1):
                    records.append(dict(name=name, seed=seed, mode=mode, batch_size=batch_size,
                                        samples=count, repeat=repeat, seconds=seconds,
                                        ms_per_image=1000*seconds/count, images_per_second=count/seconds))
            print(f'Benchmarked {name} seed={seed}', flush=True)
    raw = pd.DataFrame(records)
    raw.to_csv(OUT / 'benchmark_raw.csv', index=False)
    # First reduce timing repeats within a checkpoint, then average across seeds.
    per_run = raw.groupby(['name', 'seed', 'mode']).agg(
        ms_median=('ms_per_image', 'median'), ms_p25=('ms_per_image', lambda a: a.quantile(.25)),
        ms_p75=('ms_per_image', lambda a: a.quantile(.75))).reset_index()
    per_run.to_csv(OUT / 'benchmark_runs.csv', index=False)
    summary = per_run.groupby(['name', 'mode']).agg(
        runs=('seed', 'count'), ms_mean=('ms_median', 'mean'), ms_std=('ms_median', 'std')).reset_index()
    summary.to_csv(OUT / 'benchmark_summary.csv', index=False)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                    processor=platform.processor(), python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, threads=cfg['threads'], threadpools=pools,
                    repeats=REPEATS, warmup_batches=WARMUPS, checkpoint_sha256=files,
                    protocol='CPU float32 inputs [0,1] in RAM; fixed normalization for neural models; '
                    'predict_proba/softmax + argmax included; no loading, segmentation, IO or browser. '
                    'Single: first 100 test images in 100 calls; batch: all 10000 in batches of 256. '
                    'All outputs checked against frozen predictions before timing. Fixed run order; '
                    'repeats are timing measurements, not independent training runs.')
    (OUT / 'benchmark_protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
