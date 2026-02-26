#!/usr/bin/env python3
import os
import json
from PIL import Image
import cv2
import numpy as np
import pytesseract

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILES = [
    os.path.join(ROOT, 'heatmap model', 'heatmap model graphs', 'heatmap_summary_metrics.png'),
    os.path.join(ROOT, 'ulcer model', 'ulcer model graphs', 'ulcer_summary_metrics.png'),
]

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # upscale for better OCR
    scale = 2.5
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # increase contrast
    gray = cv2.equalizeHist(gray)
    # Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)
    # morphological opening to remove small noise
    kernel = np.ones((1,1), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th

results = {}
for path in FILES:
    key = os.path.basename(path)
    results[key] = {}
    try:
        proc = preprocess(path)
        # try multiple psm modes for robustness
        texts = {}
        for psm in [6, 4, 3, 11]:
            config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.%:ABCFDEGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            text = pytesseract.image_to_string(proc, config=config)
            texts[f'psm_{psm}'] = text.strip()
        results[key]['ocr_candidates'] = texts
        # try to extract common metric patterns
        # simple parse: find lines with 'Accuracy' 'Precision' 'Recall' 'F1' etc.
        parsed = {}
        for mode, text in texts.items():
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for ln in lines:
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F1']:
                    if metric.lower() in ln.lower():
                        # find number in line
                        import re
                        m = re.search(r"([0-9]*\.?[0-9]+)", ln)
                        if m:
                            parsed.setdefault(metric, set()).add(m.group(1))
        results[key]['parsed_metrics'] = {k: list(v) for k,v in parsed.items()}
        # save preprocessed image for inspection
        out_path = os.path.join(ROOT, 'tools', f'preproc_{key}')
        cv2.imwrite(out_path, proc)
        results[key]['preprocessed_image'] = out_path
    except Exception as e:
        results[key]['error'] = str(e)

out_file = os.path.join(ROOT, 'tools', 'ocr_preprocess_results.json')
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)

print('Wrote results to', out_file)
