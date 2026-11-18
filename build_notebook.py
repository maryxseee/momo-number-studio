"""Build and optionally execute Notebook from the readable percent-cell script."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def build(execute=False):
    cells, lines, markdown = [], [], False
    for line in (ROOT / 'train_digits.py').read_text(encoding='utf-8').splitlines():
        if line.startswith('# %%'):
            if lines:
                text = '\n'.join(lines).strip()
                cells.append(nbformat.v4.new_markdown_cell(text) if markdown else nbformat.v4.new_code_cell(text))
            lines = []
            markdown = '[markdown]' in line
        else:
            lines.append(line[2:] if markdown and line.startswith('# ') else ('' if markdown and line == '#' else line))
    if lines:
        text = '\n'.join(lines).strip()
        cells.append(nbformat.v4.new_markdown_cell(text) if markdown else nbformat.v4.new_code_cell(text))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
    nbformat.validate(nb)
    if execute:
        NotebookClient(nb, timeout=900, resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / 'digits_classification.ipynb')
    print(f'Notebook saved: {len(cells)} cells, executed={execute}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    build(parser.parse_args().execute)
