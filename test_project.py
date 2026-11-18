"""Acceptance checks for the executed educational experiment (stdlib only)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ProjectChecks(unittest.TestCase):
    def test_experiment_artifacts_and_integrity(self):
        path = ROOT / 'results' / 'metrics.json'
        self.assertTrue(path.exists(), 'Experiment has not produced metrics.json')
        r = json.loads(path.read_text(encoding='utf-8'))
        train, test = set(r['train_indices']), set(r['test_indices'])
        self.assertFalse(train & test)
        self.assertEqual(len(train | test), 1797)
        self.assertEqual(len(test), 360)
        self.assertEqual(r['selection_rule'], 'training_cv_macro_f1')
        self.assertEqual(r['selected_model'], max(r['cv_macro_f1'], key=r['cv_macro_f1'].get))
        self.assertEqual(sum(map(sum, r['confusion_matrix'])), len(test))
        correct = sum(r['confusion_matrix'][i][i] for i in range(10))
        self.assertAlmostEqual(r['test_accuracy'], correct / len(test))
        self.assertEqual(r['errors'], len(test) - correct)
        self.assertGreater(r['test_accuracy'], r['baseline_accuracy'])
        self.assertEqual([x['sigma'] for x in r['noise']], [0, 1, 2, 4])
        for name in ['samples', 'class_balance', 'cv_comparison', 'confusion_matrix', 'errors', 'learning_curve', 'noise_robustness']:
            self.assertGreater((ROOT / 'results' / f'{name}.png').stat().st_size, 1000)

    def test_notebook_executed_without_errors(self):
        path = ROOT / 'digits_classification.ipynb'
        self.assertTrue(path.exists(), 'Notebook is missing')
        nb = json.loads(path.read_text(encoding='utf-8'))
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertGreater(len(code), 8)
        for cell in code:
            self.assertIsNotNone(cell['execution_count'])
            self.assertFalse(any(o['output_type'] == 'error' for o in cell['outputs']))
        self.assertTrue(any(o['output_type'] == 'display_data' and 'image/png' in o.get('data', {}) for c in code for o in c['outputs']))


if __name__ == '__main__':
    unittest.main()
