"""Reproduces the CellRefine test results of YOLO26-m (three runs) on CPU or GPU from the cached detector predictions
and the released refiner weights, without the detector weights.

Run from the repository root after placing the TXL-PBC dataset in TXL-PBC_Dataset-master/TXL-PBC and downloading
the weights (python download_weights.py):
    python analysis/reproduce_cellrefine.py            # test split
    python analysis/reproduce_cellrefine.py --split val
Metrics use analysis/evalcore.py, a re-implementation of the Ultralytics 8.4.191 detection metrics.
The refined boxes are saved to results/analysis/refined_all.pkl for the per-detection analysis and the figures.
"""
import argparse, os, pickle, sys, json, warnings
from pathlib import Path
import numpy as np, torch
from PIL import Image
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis'))
os.chdir(ROOT)
from cellrefine.refiner import CellRefiner, refine_preds
import evalcore as E

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='test', choices=['test', 'val'])
ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
IMG = ROOT / 'TXL-PBC_Dataset-master' / 'TXL-PBC' / 'images'
G = E.load_gt(a.split)
out_path = ROOT / 'results' / 'analysis' / 'refined_all.pkl'
store = pickle.load(open(out_path, 'rb')) if out_path.exists() else {}
rows = []
for s in [int(x) for x in a.seeds.split(',')]:
    wf = ROOT / f'TXLPBC_runs/cellrefine/runs/r18_s{s}/best.pt'
    if not wf.exists(): sys.exit(f'{wf} not found; download the weights first: python download_weights.py')
    ck = torch.load(wf, map_location='cpu', weights_only=False)
    ar = ck['args']
    m = CellRefiner(size=ar['size'], ctx=ar['ctx'], bins=ar['bins'], rng=ar['rng'], backbone=ar['backbone'], pretrained=False)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck['model'].items()}); m = m.to(a.device).eval()
    ch = json.load(open(ROOT / f'TXLPBC_runs/cellrefine/metrics/r18_s{s}_lam.json'))['chosen']   # selected on the validation set
    dp = pickle.loads((ROOT / f'TXLPBC_runs/cellrefine/preds/yolo26m_s{s}_v_{a.split}.pkl').read_bytes())
    ims = {k: np.array(Image.open(IMG / a.split / os.path.basename(k)).convert('RGB')) for k in dp}
    rp = refine_preds(m, dp, ims, iters=ch['iters'], beta=ch['beta'], lam=ch['lam'], device=a.device)
    stem = lambda P: {os.path.basename(k)[:-4]: v for k, v in P.items()}
    r0, r1 = E.evaluate(stem(dp), G), E.evaluate(stem(rp), G)
    rows.append((s, r0['map'] * 100, r1['map'] * 100, r0['cls_map'][2] * 100, r1['cls_map'][2] * 100))
    print(f'run {s}: mAP50-95 {r0["map"] * 100:.2f} -> {r1["map"] * 100:.2f} | platelet AP50-95 '
          f'{r0["cls_map"][2] * 100:.2f} -> {r1["cls_map"][2] * 100:.2f}', flush=True)
    store[(s, a.split)] = dict(base=dp, ref=rp)
r = np.array(rows)
print(f'mean ± SD: {r[:, 1].mean():.2f} ± {r[:, 1].std(ddof=1):.2f} -> {r[:, 2].mean():.2f} ± {r[:, 2].std(ddof=1):.2f}')
pickle.dump(store, open(out_path, 'wb'))
