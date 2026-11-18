"""Acceptance checks for the complete, executed research deliverable."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactChecks(unittest.TestCase):
    def test_completed_experiment(self):
        path = ROOT / 'artifacts' / 'evaluation.json'
        self.assertTrue(path.exists(), 'Final evaluation has not run')
        result = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(result['test_samples'], 10000)
        self.assertEqual(len(result['runs']), 10)
        self.assertEqual(result['selection_rule'], 'validation_macro_f1')
        for row in result['runs']:
            self.assertEqual(len(row['conditions']), 7)
            clean = row['conditions']['clean']
            self.assertAlmostEqual(clean['accuracy'], clean['correct']/10000)
            self.assertEqual(sum(map(sum, row['confusion_matrix'])), 10000)
        for seed in [17, 42, 73]:
            a = json.loads((ROOT/'artifacts'/f'CNN_{seed}.json').read_text())
            b = json.loads((ROOT/'artifacts'/f'CNN_aug_{seed}.json').read_text())
            self.assertEqual(a['initial_weights_sha256'], b['initial_weights_sha256'])
        selected = json.loads((ROOT/'artifacts'/'demo_selection.json').read_text())
        neural = [r for r in result['runs'] if r['name'] != 'LogisticRegression']
        self.assertEqual(selected['validation_macro_f1'], max(r['validation_macro_f1'] for r in neural))

    def test_executed_notebook_and_report(self):
        path = ROOT / 'research.ipynb'
        self.assertTrue(path.exists(), 'Research Notebook is missing')
        nb = json.loads(path.read_text(encoding='utf-8'))
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertGreaterEqual(len(code), 10)
        for c in code:
            self.assertIsNotNone(c['execution_count'])
            self.assertFalse(any(o['output_type']=='error' for o in c['outputs']))
        self.assertGreater((ROOT/'REPORT.md').stat().st_size, 6000)


if __name__ == '__main__':
    unittest.main()
