import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class NumberArtifactChecks(unittest.TestCase):
    def test_real_benchmark_results(self):
        path=ROOT/'artifacts'/'numbers'/'summary.json'
        self.assertTrue(path.exists(),'Synthetic number benchmark has not run')
        result=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(result['source'],'official_mnist_test_digits_synthetic_composition')
        for layout in ['spaced','zero_gap']:
            r=result['layouts'][layout]
            self.assertEqual(r['samples'],1000)
            self.assertEqual(r['numbers_min'],1)
            self.assertEqual(r['numbers_max'],1000)
            self.assertAlmostEqual(r['exact_number_accuracy'],r['exact_numbers']/1000)
            self.assertEqual(r['oracle_digits_total'],2893)
        self.assertTrue((ROOT/'artifacts'/'numbers'/'examples.png').exists())


if __name__=='__main__':
    unittest.main()
