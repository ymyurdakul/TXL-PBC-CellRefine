"""Collects the CellRefine results into one table: TXLPBC_runs/cellrefine/cellrefine_summary.csv"""
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
    rows.append({"file": f.split("/")[-1][:-5], "detector": j["det"], "refiner": j["run"], "tta": j.get("tta", 1),
                 "selected": "i%s_l%s" % (j["chosen"]["iters"], j["chosen"].get("lam", 1.0)),
                 "val_before": round(j["base_val"]["mAP50-95"], 4), "val_after": round(j["val"]["mAP50-95"], 4),
                 "test_before": round(b["mAP50-95"], 4), "test_after": round(t["mAP50-95"], 4),
                 "gain": round(t["mAP50-95"] - b["mAP50-95"], 4),
                 "test_mAP50_before": round(b["mAP50"], 4), "test_mAP50_after": round(t["mAP50"], 4),
                 "test_mAP75_before": round(b["mAP75"], 4), "test_mAP75_after": round(t["mAP75"], 4),
                 **{f"{c}_before": round(b[f"AP50-95_{c}"], 4) for c in ("WBC", "RBC", "Platelets")},
                 **{f"{c}_after": round(t[f"AP50-95_{c}"], 4) for c in ("WBC", "RBC", "Platelets")},
                 "refiner_M": j["refiner_params_M"], "ms_per_image": j["refine_ms_per_img"]})
with open(OUT / "cellrefine_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print("%-28s %-12s %-8s val %.4f->%.4f  test %.4f->%.4f (%+.4f)  mAP75 %.4f->%.4f  W %.3f->%.3f R %.3f->%.3f P %.3f->%.3f" % (
        r["file"], r["detector"], r["selected"], r["val_before"], r["val_after"], r["test_before"], r["test_after"], r["gain"],
        r["test_mAP75_before"], r["test_mAP75_after"], r["WBC_before"], r["WBC_after"], r["RBC_before"], r["RBC_after"],
        r["Platelets_before"], r["Platelets_after"]))
main = [r for r in rows if r["file"] in ("r18_s0_lam", "r18_s1_lam", "r18_s2_lam")]
if main:
    for k in ("test_before", "test_after", "val_before", "val_after", "Platelets_before", "Platelets_after"):
        v = np.array([r[k] for r in main])
        print("MAIN (YOLO26-m, %d runs) %s: %.4f +- %.4f" % (len(v), k, v.mean(), v.std(ddof=1) if len(v) > 1 else 0))
