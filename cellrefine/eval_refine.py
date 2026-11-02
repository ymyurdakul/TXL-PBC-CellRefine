"""CellRefine evaluation: the settings (iters, beta, lam) are selected on the validation set only, and the test set
is evaluated a single time. Example: python cellrefine/eval_refine.py --det yolo26m_s0 --run r18_s0
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellrefine.common import OUT, cached_preds, evaluate, image_list, load_gt
from cellrefine.refiner import CellRefiner, refine_preds

ap = argparse.ArgumentParser()
ap.add_argument("--det", default="yolo26m_s0")
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--iters", default="0,1,2,3")
ap.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
ap.add_argument("--tag", default="")
ap.add_argument("--tta", type=int, default=1)
ap.add_argument("--lams", default="1.0")
a = ap.parse_args()
from functools import partial
refine_preds = partial(refine_preds, tta=a.tta)


def load_split(split):
    ims, gts = {}, {}
    for p in image_list(split):
        im = np.array(Image.open(p).convert("RGB"))
        ims[p] = im
        gts[p] = load_gt(p, (im.shape[1], im.shape[0]))
    return ims, gts


ck = torch.load(OUT / "runs" / a.run / a.ckpt, map_location="cpu", weights_only=False)
ar = ck["args"]
model = CellRefiner(size=ar["size"], ctx=ar["ctx"], bins=ar["bins"], rng=ar["rng"], backbone=ar["backbone"],
                    pretrained=False)
model.load_state_dict(ck["model"])
model = model.cuda().eval()
params = sum(p.numel() for p in model.parameters()) / 1e6

res = {"tta": a.tta, "det": a.det, "run": a.run, "ckpt": a.ckpt, "ep": ck.get("ep"), "refiner_params_M": round(params, 3)}
data = {s: load_split(s) for s in ("val", "test")}
dets = {s: cached_preds(a.det, s) for s in ("val", "test")}
for s in ("val", "test"):
    res[f"base_{s}"] = evaluate(dets[s], data[s][1])

grid = {}
for it in [int(x) for x in a.iters.split(",")]:
    for beta in [float(x) for x in a.betas.split(",")]:
        for lam in ([1.0] if it == 0 else [float(x) for x in a.lams.split(",")]):
            r = evaluate(refine_preds(model, dets["val"], data["val"][0], iters=it, beta=beta, lam=lam), data["val"][1])
            grid[(it, beta, lam)] = r["mAP50-95"]
            print(f"val iters={it} beta={beta} lam={lam}: {r['mAP50-95']:.4f}", flush=True)
it, beta, lam = max(grid, key=grid.get)
key = f"i{it}_b{beta}_l{lam}"
refine_preds = partial(refine_preds, lam=lam)
res["val_grid"] = {f"i{k[0]}_b{k[1]}_l{k[2]}": v for k, v in grid.items()}
res["chosen"] = {"iters": it, "beta": beta, "lam": lam}

torch.cuda.synchronize()
t0 = time.time()
ref_test = refine_preds(model, dets["test"], data["test"][0], iters=it, beta=beta)
torch.cuda.synchronize()
n_box = sum(len(v["box"]) for v in dets["test"].values())
res["refine_ms_per_img"] = round((time.time() - t0) * 1000 / len(dets["test"]), 2)
res["boxes_per_img"] = round(n_box / len(dets["test"]), 1)
res["val"] = evaluate(refine_preds(model, dets["val"], data["val"][0], iters=it, beta=beta), data["val"][1])
res["test"] = evaluate(ref_test, data["test"][1])
# component analysis (with the same selected settings)
res["test_rescore_only"] = evaluate(refine_preds(model, dets["test"], data["test"][0], iters=0, beta=beta),
                                    data["test"][1])
res["test_refine_only"] = evaluate(refine_preds(model, dets["test"], data["test"][0], iters=max(it, 1), beta=0.0),
                                   data["test"][1])
d = OUT / "metrics"
d.mkdir(parents=True, exist_ok=True)
(d / f"{a.run}{a.tag}.json").write_text(json.dumps(res, indent=2))
b, t = res["base_test"], res["test"]
print(f"SELECTED {key} | val {res['base_val']['mAP50-95']:.4f} -> {res['val']['mAP50-95']:.4f} | test "
      f"{b['mAP50-95']:.4f} -> {t['mAP50-95']:.4f} (rescoring only {res['test_rescore_only']['mAP50-95']:.4f}, "
      f"refinement only {res['test_refine_only']['mAP50-95']:.4f})", flush=True)
for c in ("WBC", "RBC", "Platelets"):
    print(f"  {c}: {b[f'AP50-95_{c}']:.4f} -> {t[f'AP50-95_{c}']:.4f}")
