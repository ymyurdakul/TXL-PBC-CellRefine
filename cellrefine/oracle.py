"""Basari tavani analizi: (a) yeniden puanlama kahini, (b) konumlama kahini. Tahminleri onbellege yazar."""
import pickle
import sys

import numpy as np

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellrefine.common import DETECTORS, OUT, box_iou, evaluate, gt_dict, predict


def cached_preds(det, split):
    f = OUT / "preds" / f"{det}_{split}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    f.parent.mkdir(parents=True, exist_ok=True)
    p = predict(DETECTORS[det], split)
    f.write_bytes(pickle.dumps(p))
    return p


def oracle(preds, gts, mode):
    out = {}
    for k, p in preds.items():
        gb, gc = gts[k]
        q = {kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in p.items()}
        if len(p["box"]) and len(gb):
            iou = box_iou(p["box"], gb) * (p["cls"][:, None] == gc[None, :])
            j = iou.argmax(1)
            best = iou.max(1)
            if mode == "score":
                q["conf"] = best.astype(np.float32)
            elif mode == "score_mix":
                q["conf"] = (p["conf"] * best).astype(np.float32)
            elif mode == "loc":
                m = best >= 0.5
                q["box"][m] = gb[j[m]]
        out[k] = q
    return out


if __name__ == "__main__":
    det = sys.argv[1] if len(sys.argv) > 1 else "yolo26m_s0"
    for split in ("val", "test"):
        P = cached_preds(det, split)
        G = gt_dict(P)
        n = sum(len(v["box"]) for v in P.values())
        print(f"{det} {split}: {len(P)} goruntu, {n} tahmin")
        for mode in ("none", "score", "score_mix", "loc"):
            r = evaluate(oracle(P, G, mode) if mode != "none" else P, G)
            print(f"  {mode:10s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
