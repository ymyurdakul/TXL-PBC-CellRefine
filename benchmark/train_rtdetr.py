# Training and evaluation of the non-YOLO detectors in Ultralytics (RT-DETR-l, RT-DETR-x) with three seeds.
# Same settings as the YOLO runs: epochs=100, imgsz=640, batch=16, workers=2, cache=ram, deterministic, optimizer=auto.
# seed 0   -> train/, metrics/, eval_val/, eval_test/, predict_test/ (same layout as train_all_yolo.py)
# seed 1,2 -> seeds/train/, metrics_seeds/, seeds/eval_*/ (same layout as seeds_all.py)
# Run: nohup python benchmark/train_rtdetr.py > rtdetr_all.log 2>&1 &   (finished runs are skipped on a rerun)
import os, sys, json, time, csv, traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from pathlib import Path

import pandas as pd
import torch
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import get_flops

ROOT = (Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1])) / "TXL-PBC_Dataset-master/TXL-PBC")
OUT = (Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1])) / "TXLPBC_runs")
DATA = OUT / "txlpbc.yaml"
NAMES = ["WBC", "RBC", "Platelets"]
EPOCHS, IMGSZ, BATCH = 100, 640, 16
MODELS = ["rtdetr-l.pt", "rtdetr-x.pt"]
SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
(OUT / "metrics_seeds").mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def metrics_of(r):
    b = r.box
    d = {"P": float(b.mp), "R": float(b.mr), "mAP50": float(b.map50), "mAP75": float(b.map75),
         "mAP50-95": float(b.map), "fitness": float(r.fitness)}
    for i, n in enumerate(NAMES):
        try:
            idx = list(b.ap_class_index).index(i)
            d[f"P_{n}"] = float(b.p[idx]); d[f"R_{n}"] = float(b.r[idx])
            d[f"AP50_{n}"] = float(b.ap50[idx]); d[f"AP50-95_{n}"] = float(b.ap[idx])
        except ValueError:
            pass
    d.update({f"{k}_ms": float(v) for k, v in r.speed.items()})
    return d


def write_summary():  # same format as the summary in train_all_yolo.py
    rows = []
    for f in sorted((OUT / "metrics").glob("*.json")):
        j = json.loads(f.read_text())
        for split in ("val", "test"):
            if split in j:
                rows.append({"model": j["model"], "split": split, "pretrained": j["pretrained"],
                             "params_M": j.get("params_M"), "GFLOPs": j.get("GFLOPs"),
                             "train_time_h": j.get("train_time_h"), **j[split]})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / "summary_all_models.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


for srcw in MODELS:
    m = Path(srcw).stem
    for s in SEEDS:
        main = s == 0
        name = m if main else f"{m}_s{s}"
        base = OUT if main else OUT / "seeds"
        mf = (OUT / "metrics" / f"{m}.json") if main else (OUT / "metrics_seeds" / f"{name}.json")
        if mf.exists():
            log(f"{name}: already finished, skipped"); continue
        run = base / "train" / name
        last = run / "weights" / "last.pt"
        batch = BATCH
        for attempt in range(3):
            try:
                t0 = time.time()
                if (run / "TRAINED").exists():
                    log(f"{name}: training finished, evaluation only")
                elif last.exists():
                    log(f"{name}: resume"); RTDETR(str(last)).train(resume=True)
                else:
                    log(f"{name}: training (pretrained=True, seed={s}, batch={batch})")
                    RTDETR(srcw).train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=batch, device=0,
                                       workers=2, cache="ram", seed=s, deterministic=True, pretrained=True,
                                       project=str(base / "train"), name=name, exist_ok=True, plots=main,
                                       save=True, val=True, verbose=main)
                (run / "TRAINED").touch()
                rc = pd.read_csv(run / "results.csv"); rc.columns = rc.columns.str.strip()
                train_h = float(rc["time"].iloc[-1]) / 3600 if "time" in rc else (time.time() - t0) / 3600
                best = RTDETR(str(run / "weights" / "best.pt"))
                res = {"model": m, "source": srcw, "seed": s, "pretrained": True, "epochs": EPOCHS,
                       "batch": batch, "family": "RT-DETR",
                       "params_M": round(sum(p.numel() for p in best.model.parameters()) / 1e6, 3),
                       "GFLOPs": round(get_flops(best.model, IMGSZ) or 0.0, 2), "train_time_h": round(train_h, 3)}
                for split in ("val", "test"):
                    kw = dict(plots=True, save_json=True, save_txt=True, save_conf=True) if main else \
                         dict(plots=False, save_json=(split == "test"), verbose=False)
                    r = best.val(data=str(DATA), split=split, imgsz=IMGSZ, batch=BATCH, device=0,
                                 project=str(base / f"eval_{split}"), name=name, exist_ok=True, **kw)
                    res[split] = metrics_of(r)
                if main:
                    best.predict(source=str(ROOT / "images" / "test"), imgsz=IMGSZ, device=0, conf=0.25,
                                 project=str(OUT / "predict_test"), name=name, exist_ok=True,
                                 save=True, save_txt=True, save_conf=True, verbose=False)
                mf.write_text(json.dumps(res, indent=2))
                if main:
                    write_summary(); (run / "DONE").touch()
                elif last.exists():
                    last.unlink()  # save disk space; best.pt is kept
                log(f"{name}: DONE val={res['val']['mAP50-95']:.4f} test={res['test']['mAP50-95']:.4f}")
                break
            except Exception as e:
                log(f"{name}: ERROR (attempt {attempt + 1}) {e}")
                with open(OUT / "errors_rtdetr.log", "a") as fh:
                    fh.write(f"\n===== {name} {time.ctime()} =====\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower() and not last.exists() and batch > 8:
                    batch = 8; log(f"{name}: OOM -> retry with batch=8")
                elif "out of memory" in str(e).lower():
                    time.sleep(300)
                else:
                    break
            finally:
                torch.cuda.empty_cache()
log("All RT-DETR runs finished.")
