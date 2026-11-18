from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class NumberChecks(unittest.TestCase):
    def test_segmentation_order_and_noise(self):
        import inference
        self.assertTrue(hasattr(inference,'segment_number'),'Number segmentation is missing')
        image=np.zeros((240,840),dtype=np.uint8)
        image[60:190,70:90]=255
        image[60:190,210:260]=255
        image[60:190,264:280]=255  # detached stroke, only 4 blank columns
        image[2,700]=255  # isolated speck must not become a digit
        parts=inference.segment_number(image)
        self.assertEqual(len(parts),2)
        self.assertEqual(parts[0]['bbox'],[70,60,20,130])
        self.assertEqual(parts[1]['bbox'],[210,60,70,130])
        with self.assertRaises(ValueError):
            inference.segment_number(np.zeros((240,840)))
        many=np.zeros((240,840),dtype=np.uint8)
        for i in range(5):
            many[60:190,30+120*i:50+120*i]=255
        with self.assertRaises(ValueError):
            inference.segment_number(many)

    def test_strict_range(self):
        import inference
        self.assertTrue(hasattr(inference,'validate_number'),'Range validation is missing')
        for text,value in [('1',1),('407',407),('999',999),('1000',1000)]:
            self.assertEqual(inference.validate_number(text),(value,None))
        for text in ['0','1001','9999','01','0001','', '12345']:
            number,error=inference.validate_number(text)
            self.assertIsNone(number)
            self.assertTrue(error)

    def test_actual_number_inference(self):
        import inference
        self.assertTrue(hasattr(inference,'predict_number'),'Number inference is missing')
        import json
        import torch
        torch.set_num_threads(2)
        selected=json.loads((ROOT/'artifacts'/'demo_selection.json').read_text())
        model,_=inference.load_checkpoint(ROOT/'artifacts'/selected['checkpoint'])
        image=np.zeros((240,840),dtype=np.uint8)
        image[50:190,100:116]=255
        result=inference.predict_number(model,image)
        self.assertEqual(result['text'],'1')
        self.assertEqual(result['number'],1)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['digits']),1)
        self.assertEqual(len(result['digits'][0]['preview']),28)


if __name__=='__main__':
    unittest.main()
