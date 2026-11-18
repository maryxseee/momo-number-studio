"""Copy the tested static app to the GitHub Pages /docs source."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def build():
    source=ROOT/'site';destination=ROOT/'docs'
    for required in ['index.html','style.css','app.mjs','engine.mjs','assets/digit-cnn.onnx','vendor/ort-wasm-simd-threaded.wasm']:
        if not (source/required).is_file():
            raise FileNotFoundError(required)
    # Preserve unrelated local planning notes, which are ignored by Git.
    for path in source.rglob('*'):
        if not path.is_file() or 'tests' in path.relative_to(source).parts:
            continue
        target=destination/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    manifest={p.relative_to(source).as_posix():p.stat().st_size for p in source.rglob('*') if p.is_file() and 'tests' not in p.parts}
    print(json.dumps({'pages_source':'main:/docs','files':len(manifest),'bytes':sum(manifest.values())}))


if __name__=='__main__':
    build()
