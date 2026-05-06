import json
for path in ['notebooks/train_model_b_perch.ipynb', 'notebooks/evaluate_best.ipynb']:
    nb = json.load(open(path, encoding='utf-8'))
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        joined = ''.join(cell['source'])
        if 'birdclef2026-code' not in joined: continue
        if 'datasets/*/birdclef2026-code' in joined:
            continue  # already patched
        new_src = []
        for line in cell['source']:
            if "'/kaggle/input/*birdclef2026-code*'" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_src.append(f"{indent}'/kaggle/input/datasets/*/birdclef2026-code',\n")
                new_src.append(f"{indent}'/kaggle/input/*/birdclef2026-code',\n")
                changed = True
            new_src.append(line)
        cell['source'] = new_src
    if changed:
        json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1)
        print('patched', path)
    else:
        print('no change', path)
