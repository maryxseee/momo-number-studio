"""Bundle the project, executed outputs and all checkpoints, without dependencies/data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
EXCLUDED={'data','.runtime','.venv','__pycache__','.ipynb_checkpoints'}


def package():
    files=[p for p in ROOT.rglob('*') if p.is_file() and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
           and p.name!='release_manifest.json']
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest_path=ROOT/'artifacts'/'release_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    destination=ROOT.parent/'mnist_research_project.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files+[manifest_path]):
            z.write(p,Path('mnist_project')/p.relative_to(ROOT))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('.pt')])==9
        assert 'mnist_project/research.ipynb' in z.namelist()
        for name,checksum in manifest.items():
            assert hashlib.sha256(z.read('mnist_project/'+name)).hexdigest()==checksum
    print(f'ZIP verified: {destination.name}, {destination.stat().st_size:,} bytes, {len(files)+1} files')


if __name__=='__main__':
    package()
