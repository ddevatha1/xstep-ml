#!/usr/bin/env python3
import os
import re
from PIL import Image
import pytesseract

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FOLDERS = [
    os.path.join(ROOT, 'heatmap model', 'heatmap model graphs'),
    os.path.join(ROOT, 'ulcer model', 'ulcer model graphs')
]

number_re = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?%?")
auc_re = re.compile(r"AUC\s*=\s*([0-9]*\.?[0-9]+)")
metric_re = re.compile(r"(Accuracy|Precision|Recall|F1|F1-Score)[:\s]+([0-9]*\.?[0-9]+)")

results = {}

for folder in FOLDERS:
    if not os.path.isdir(folder):
        continue
    results[folder] = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.png'):
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
        except Exception as e:
            results[folder][fname] = {'error': str(e)}
            continue
        text = pytesseract.image_to_string(img)
        nums = number_re.findall(text)
        aucs = auc_re.findall(text)
        metrics = metric_re.findall(text)
        # also try to parse small table like 'Accuracy: 0.8234' or 'Accuracy  0.8234'
        table_metrics = {m[0]: m[1] for m in metrics}
        results[folder][fname] = {
            'raw_text': text.strip(),
            'numbers_found': nums,
            'aucs': aucs,
            'metrics': table_metrics,
        }

# Print concise summary
for folder, files in results.items():
    print('\nFolder:', folder)
    for fname, info in files.items():
        print('\nFile:', fname)
        if 'error' in info:
            print('  Error reading file:', info['error'])
            continue
        print('  AUCs found:', info['aucs'] if info['aucs'] else 'none')
        if info['metrics']:
            for k, v in info['metrics'].items():
                print(f'  {k}: {v}')
        else:
            # if no named metrics, show top numeric candidates
            nums = info['numbers_found']
            if nums:
                # show first 10
                print('  Numbers (sample):', ', '.join(nums[:10]))
            else:
                print('  No numeric text detected')

# Optionally write full OCR outputs
out_path = os.path.join(ROOT, 'tools', 'ocr_extraction_results.txt')
with open(out_path, 'w') as f:
    import json
    json.dump(results, f, indent=2)

print('\nFull OCR output written to', out_path)
