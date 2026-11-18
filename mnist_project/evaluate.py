"""Apply the frozen evaluation protocol after training is complete."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data import load_test, corrupt, CONDITIONS
from inference import load_checkpoint
from models import normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def neural_predictions(model, x):
    predictions, confidence = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            probabilities = model(normalize(x[start:start+512])).softmax(1)
            score, label = probabilities.max(1)
            predictions.extend(label.tolist())
            confidence.extend(score.tolist())
    return np.array(predictions), np.array(confidence)


def run():
    cfg = json.loads((OUT / 'protocol.json').read_text())
    if cfg['conditions'] != CONDITIONS:
        raise ValueError('Evaluation conditions differ from the frozen protocol')
    torch.set_num_threads(cfg['threads'])
    training = pd.read_csv(OUT / 'training_summary.csv')
    if len(training) != 3 * len(cfg['seeds']) + 1:
        raise ValueError('Training is incomplete')
    x, y = load_test()
    truth = y.numpy()
    inputs = {condition: corrupt(x, condition) for condition in CONDITIONS}
    np.savez_compressed(OUT / 'corruption_examples.npz', **{k: v[:10].numpy() for k, v in inputs.items()}, labels=truth[:10])
    records, table, saved_predictions = [], [], {'truth': truth}
    for metadata in training.to_dict('records'):
        name, seed = metadata['name'], int(metadata['seed'])
        key = f'{name}_{seed}'
        if name == 'LogisticRegression':
            model = joblib.load(OUT / 'logistic_regression.joblib')
            model_path = OUT / 'logistic_regression.joblib'
        else:
            model_path = OUT / metadata['checkpoint']
            model, _ = load_checkpoint(model_path)
        record = {'name': name, 'seed': seed, 'validation_macro_f1': metadata['validation_macro_f1'],
                  'checkpoint_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'conditions': {}}
        for condition, images in inputs.items():
            start = time.perf_counter()
            if name == 'LogisticRegression':
                probs = model.predict_proba(images.flatten(1).numpy())
                prediction, confidence = probs.argmax(1), probs.max(1)
            else:
                prediction, confidence = neural_predictions(model, images)
            seconds = time.perf_counter() - start
            correct = int((prediction == truth).sum())
            ci = binomtest(correct, len(truth)).proportion_ci(0.95, method='wilson')
            metrics = {'accuracy': accuracy_score(truth, prediction), 'macro_f1': f1_score(truth, prediction, average='macro'),
                       'correct': correct, 'accuracy_wilson_95': [ci.low, ci.high], 'inference_seconds': seconds}
            record['conditions'][condition] = metrics
            table.append({'name': name, 'seed': seed, 'condition': condition, **{k:v for k,v in metrics.items() if k != 'accuracy_wilson_95'}})
            saved_predictions[f'{key}__{condition}'] = prediction.astype(np.int8)
            saved_predictions[f'{key}__{condition}__confidence'] = confidence.astype(np.float32)
            if condition == 'clean':
                record['confusion_matrix'] = confusion_matrix(truth, prediction).tolist()
                pd.DataFrame(classification_report(truth, prediction, output_dict=True, zero_division=0)).T.to_csv(OUT / f'classes_{key}.csv')
        records.append(record)
        print(f'{key}: test accuracy={record["conditions"]["clean"]["accuracy"]:.4f}', flush=True)
    np.savez_compressed(OUT / 'predictions.npz', **saved_predictions)
    results = pd.DataFrame(table)
    results.to_csv(OUT / 'evaluation.csv', index=False)
    aggregated = results.groupby(['name', 'condition']).agg(n=('seed', 'count'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std')).reset_index()
    aggregated.to_csv(OUT / 'aggregated.csv', index=False)
    paired = results[results.name=='CNN_aug'].merge(results[results.name=='CNN'], on=['seed', 'condition'], suffixes=('_aug', '_plain'))
    paired['delta_accuracy'] = paired.accuracy_aug - paired.accuracy_plain
    paired['delta_macro_f1'] = paired.macro_f1_aug - paired.macro_f1_plain
    paired[['seed', 'condition', 'delta_accuracy', 'delta_macro_f1']].to_csv(OUT / 'paired_deltas.csv', index=False)
    summary = {'test_samples': len(y), 'selection_rule': 'validation_macro_f1', 'runs': records,
               'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()}
    (OUT / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Evaluation complete. Parameters remain frozen.', flush=True)


if __name__ == '__main__':
    run()
