"""Local-only drawing demo. No cloud service, uploads or image persistence."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch
from inference import load_checkpoint, predict_canvas, predict_number

ROOT = Path(__file__).resolve().parent


def make_handler(model, metadata):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/health':
                return self.send_json(200, {'ok': True, 'model': metadata['name'], 'seed': metadata['seed'], 'epoch': metadata['epoch']})
            if self.path not in ['/','/digit']:
                return self.send_json(404, {'error': 'Not found'})
            filename='index.html' if self.path=='/digit' else 'numbers.html'
            raw = (ROOT / 'web' / filename).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            if self.path not in ['/predict','/predict-number']:
                return self.send_json(404, {'error': 'Not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1500000:
                    raise ValueError('Завеликий або порожній запит.')
                body = json.loads(self.rfile.read(size))
                predictor=predict_number if self.path=='/predict-number' else predict_canvas
                result = predictor(model, body['pixels'])
                self.send_json(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': str(exc)})

    return Handler


def serve(port=8765):
    torch.set_num_threads(2)
    selected = json.loads((ROOT/'artifacts'/'demo_selection.json').read_text())
    model, metadata = load_checkpoint(ROOT/'artifacts'/selected['checkpoint'])
    server = HTTPServer(('127.0.0.1', port), make_handler(model, metadata))
    print(f'Відкрийте http://127.0.0.1:{port} — {metadata["name"]}, seed={metadata["seed"]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
