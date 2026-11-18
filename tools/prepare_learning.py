"""Vendor the pinned TensorFlow.js library used for local last-layer training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def run():
    metadata=json.load(urllib.request.urlopen('https://registry.npmjs.org/@tensorflow%2ftfjs/4.22.0'))
    raw=urllib.request.urlopen(metadata['dist']['tarball']).read()
    integrity='sha512-'+base64.b64encode(hashlib.sha512(raw).digest()).decode()
    if integrity!=metadata['dist']['integrity']:
        raise ValueError('TensorFlow.js package integrity mismatch')
    destination=ROOT/'site'/'vendor'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        (destination/'tf.min.js').write_bytes(archive.extractfile('package/dist/tf.min.js').read())
    (destination/'TENSORFLOW-LICENSE').write_bytes(urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt').read())
    (destination/'tensorflow-version.json').write_text(json.dumps({'package':'@tensorflow/tfjs','version':'4.22.0','integrity':integrity,'source':metadata['dist']['tarball']},indent=2))
    print('Verified and vendored TensorFlow.js4.22.0')


if __name__=='__main__':
    run()
