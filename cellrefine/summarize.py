"""CellRefine sonuclarini tabloya doker: TXLPBC_runs/cellrefine/ozet_cellrefine.csv"""
import csv
import glob
import json
import sys

import numpy as np

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellrefine.common import OUT

rows = []
for f in sorted(glob.glob(str(OUT / "metrics" / "*.json"))):
    j = json.load(open(f))
    b, t = j["base_test"], j["test"]
    rows.append({"dosya": f.split("/")[-1][:-5], "dedektor": j["det"], "refiner": j["run"], "tta": j.get("tta", 1),
                 "secim": "i%s_l%s" % (j["chosen"]["iters"], j["chosen"].get("lam", 1.0)),
                 "val_once": round(j["base_val"]["mAP50-95"], 4), "val_sonra": round(j["val"]["mAP50-95"], 4),
                 "test_once": round(b["mAP50-95"], 4), "test_sonra": round(t["mAP50-95"], 4),
                 "fark": round(t["mAP50-95"] - b["mAP50-95"], 4),
                 "test_mAP50_once": round(b["mAP50"], 4), "test_mAP50_sonra": round(t["mAP50"], 4),
                 "test_mAP75_once": round(b["mAP75"], 4), "test_mAP75_sonra": round(t["mAP75"], 4),
                 **{f"{c}_once": round(b[f"AP50-95_{c}"], 4) for c in ("WBC", "RBC", "Platelets")},
                 **{f"{c}_sonra": round(t[f"AP50-95_{c}"], 4) for c in ("WBC", "RBC", "Platelets")},
                 "refiner_M": j["refiner_params_M"], "ms_goruntu": j["refine_ms_per_img"]})
with open(OUT / "ozet_cellrefine.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print("%-28s %-12s %-8s val %.4f->%.4f  test %.4f->%.4f (%+.4f)  mAP75 %.4f->%.4f  W %.3f->%.3f R %.3f->%.3f P %.3f->%.3f" % (
        r["dosya"], r["dedektor"], r["secim"], r["val_once"], r["val_sonra"], r["test_once"], r["test_sonra"], r["fark"],
        r["test_mAP75_once"], r["test_mAP75_sonra"], r["WBC_once"], r["WBC_sonra"], r["RBC_once"], r["RBC_sonra"],
        r["Platelets_once"], r["Platelets_sonra"]))
main = [r for r in rows if r["dosya"] in ("r18_s0_lam", "r18_s1_lam", "r18_s2_lam")]
if main:
    for k in ("test_once", "test_sonra", "val_once", "val_sonra", "Platelets_once", "Platelets_sonra"):
        v = np.array([r[k] for r in main])
        print("ANA (yolo26m, %d tohum) %s: %.4f +- %.4f" % (len(v), k, v.mean(), v.std(ddof=1) if len(v) > 1 else 0))
