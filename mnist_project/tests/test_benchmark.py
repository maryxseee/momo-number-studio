"""Validate benchmark provenance and aggregation, without asserting hardware speeds."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'


class BenchmarkChecks(unittest.TestCase):
    def test_raw_measurements_and_checkpoint_integrity(self):
        protocol = json.loads((OUT/'benchmark_protocol.json').read_text())
        raw = pd.read_csv(OUT/'benchmark_raw.csv')
        self.assertEqual(len(raw), 10*2*7)
        self.assertTrue((raw.seconds > 0).all())
        for _, rows in raw.groupby(['name', 'seed', 'mode']):
            self.assertEqual(set(rows.repeat), set(range(1, 8)))
        np.testing.assert_allclose(raw.ms_per_image, raw.seconds*1000/raw.samples)
        for path, digest in protocol['checkpoint_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest)
        measured = raw.groupby(['name','seed','mode']).ms_per_image.median().sort_index()
        summary = pd.read_csv(OUT/'benchmark_runs.csv').set_index(['name','seed','mode']).sort_index()
        np.testing.assert_allclose(measured, summary.ms_median)

    def test_comparison_matches_original_quality_and_training(self):
        comparison = pd.read_csv(OUT/'model_comparison.csv').set_index('model')
        self.assertEqual(set(comparison.index), {'LogisticRegression', 'MLP', 'CNN'})
        training = pd.read_csv(OUT/'training_summary.csv')
        quality = pd.read_csv(OUT/'clean_summary.csv').set_index('name')
        for name, row in comparison.iterrows():
            self.assertAlmostEqual(row.accuracy_pct, quality.loc[name,'accuracy_mean']*100)
            self.assertAlmostEqual(row.train_seconds_mean, training[training.name==name].training_seconds.mean())
        self.assertLess(comparison.loc['CNN','parameters'], comparison.loc['MLP','parameters'])
        self.assertGreater(comparison.loc['CNN','macs_per_image'], comparison.loc['MLP','macs_per_image'])


if __name__ == '__main__':
    unittest.main()
