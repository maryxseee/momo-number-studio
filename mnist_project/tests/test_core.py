import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CoreChecks(unittest.TestCase):
    def test_model_logits(self):
        self.assertTrue((ROOT / 'models.py').exists(), 'Models are not implemented')
        import torch
        from models import make_model
        for kind in ['mlp', 'cnn']:
            model = make_model(kind)
            result = model(torch.zeros(4, 1, 28, 28))
            self.assertEqual(tuple(result.shape), (4, 10))
            self.assertTrue(torch.isfinite(result).all())

    def test_split_and_augmentation(self):
        self.assertTrue((ROOT / 'data.py').exists(), 'Data pipeline is not implemented')
        import torch
        import numpy as np
        from data import split_indices, augment_batch, corrupt
        labels = np.tile(np.arange(10), 100)
        a, b = split_indices(labels, validation_size=200)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a) + len(b), len(labels))
        a2, b2 = split_indices(labels, validation_size=200)
        np.testing.assert_array_equal(a, a2)
        x = torch.rand(5, 1, 28, 28)
        for changed in [augment_batch(x), *[corrupt(x, c) for c in ['clean', 'rotate_plus20', 'rotate_minus20', 'shift_right3', 'noise_020', 'thick', 'thin']]]:
            self.assertEqual(changed.shape, x.shape)
            self.assertTrue(torch.isfinite(changed).all())
            self.assertTrue(((changed >= 0) & (changed <= 1)).all())
        self.assertTrue(torch.equal(corrupt(x, 'clean'), x))

    def test_canvas_preprocessing(self):
        self.assertTrue((ROOT / 'inference.py').exists(), 'Inference is not implemented')
        import numpy as np
        from inference import prepare_canvas
        with self.assertRaises(ValueError):
            prepare_canvas(np.zeros((280, 280)))
        with self.assertRaises(ValueError):
            prepare_canvas(np.full((28, 28), np.nan))
        with self.assertRaises(ValueError):
            prepare_canvas(np.zeros((28, 28, 3)))
        image = np.zeros((280, 280), dtype=np.float32)
        image[60:220, 130:150] = 255
        output = prepare_canvas(image)
        self.assertEqual(output.shape, (1, 1, 28, 28))
        self.assertGreater(float(output.max()), 0)
        self.assertLessEqual(float(output.max()), 1)


if __name__ == '__main__':
    unittest.main()
