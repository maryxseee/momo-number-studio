"""Static preview with explicit MIME types (Windows may label .mjs as text/plain)."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


class SiteHandler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.mjs':'application/javascript',
                    '.js':'application/javascript','.wasm':'application/wasm','.onnx':'application/octet-stream'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8768)
    port=parser.parse_args().port
    root=Path(__file__).resolve().parents[1]/'site'
    print(f'MOMO preview: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),partial(SiteHandler,directory=str(root))).serve_forever()
