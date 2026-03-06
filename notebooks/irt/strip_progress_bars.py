"""Strip tqdm/progress bar output from executed Jupyter notebooks."""
import json
import sys
import re


def strip_progress_bars(path):
    with open(path) as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        outputs = cell.get('outputs', [])
        new_outputs = []
        for out in outputs:
            if out.get('output_type') == 'stream' and out.get('name') == 'stderr':
                # Filter out lines that look like tqdm progress bars
                text = out.get('text', '')
                if isinstance(text, list):
                    text = ''.join(text)
                # Remove lines with \r (carriage return) — progress bars
                lines = text.split('\n')
                filtered = [l for l in lines if '\r' not in l and '100%|' not in l and 'it/s]' not in l]
                cleaned = '\n'.join(filtered).strip()
                if cleaned:
                    out['text'] = [cleaned + '\n']
                    new_outputs.append(out)
                # else: discard entirely
            else:
                new_outputs.append(out)
        cell['outputs'] = new_outputs

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Stripped progress bars from {path}')


if __name__ == '__main__':
    for path in sys.argv[1:]:
        strip_progress_bars(path)
