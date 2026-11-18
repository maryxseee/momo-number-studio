"""Exercise the real local HTTP handler, checkpoint and preprocessing together."""
from pathlib import Path
import json
import sys
import threading
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class HTTPChecks(unittest.TestCase):
    def test_real_handler_and_inference(self):
        from app import make_handler
        from inference import load_checkpoint
        import numpy as np
        import torch
        torch.set_num_threads(2)
        selected=json.loads((ROOT/'artifacts'/'demo_selection.json').read_text())
        model,metadata=load_checkpoint(ROOT/'artifacts'/selected['checkpoint'])
        server=HTTPServer(('127.0.0.1',0),make_handler(model,metadata))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url+'/health') as response:
                self.assertTrue(json.load(response)['ok'])
            with urllib.request.urlopen(url) as response:
                self.assertIn('canvas',response.read().decode('utf-8'))
            pixels=np.zeros((280,280),dtype=int)
            pixels[60:220,130:150]=255
            request=urllib.request.Request(url+'/predict',data=json.dumps({'pixels':pixels.tolist()}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:
                body=json.load(response)
            self.assertEqual(len(body['top3']),3)
            self.assertEqual(len(body['preview']),28)
            self.assertTrue(all(0<=r['digit']<=9 and 0<=r['score']<=1 for r in body['top3']))
            self.assertGreaterEqual(body['top3'][0]['score'],body['top3'][1]['score'])
            wide=np.zeros((240,840),dtype=int)
            wide[50:190,100:116]=255
            number_request=urllib.request.Request(url+'/predict-number',data=json.dumps({'pixels':wide.tolist()}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(number_request) as response:
                number_result=json.load(response)
            self.assertEqual(number_result['number'],1)
            self.assertTrue(number_result['valid'])
            self.assertEqual(number_result['segmentation_count'],1)
            with urllib.request.urlopen(url+'/digit') as response:
                self.assertIn('Лабораторія рукописних цифр',response.read().decode('utf-8'))
            blank=urllib.request.Request(url+'/predict',data=json.dumps({'pixels':np.zeros((28,28)).tolist()}).encode())
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(blank)
            self.assertEqual(caught.exception.code,400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__=='__main__':
    unittest.main()
