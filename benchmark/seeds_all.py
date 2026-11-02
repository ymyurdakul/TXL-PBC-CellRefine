# 43 temel YOLO modelini ek tohumlarla (varsayilan seed 1,2) egitir ve degerlendirir; seed 0 = train_all_yolo.py.
# Modeller seed-0 val mAP50-95 sirasina gore islenir (yarida kalsa bile ust siradakiler tamamlanmis olur).
# Calistirma: nohup python cellyolo/seeds_all.py 1 2 > seeds_all.log 2>&1 &   (tekrar calistirinca bitenler atlanir)
import os, sys, json, time, shutil, traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

OUT = (Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1])) / "TXLPBC_runs")
DATA = OUT / "txlpbc.yaml"
SR = OUT / "seeds"
MS = OUT / "metrics_seeds"
NAMES = ["WBC", "RBC", "Platelets"]
SEEDS = [int(s) for s in sys.argv[1:]] or [1, 2]
MS.mkdir(parents=True, exist_ok=True)


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


base = [json.loads(f.read_text()) for f in (OUT / "metrics").glob("*.json")]
base.sort(key=lambda j: -j["val"]["mAP50-95"])

# yolo26m icin seed 1-2 train_cell ile ayni ayarlarla zaten egitildi -> yeniden kullan
for s in (1, 2):
    src = OUT / "metrics_cell" / f"base_yolo26m_s{s}.json"
    dst = MS / f"yolo26m_s{s}.json"
    if src.exists() and not dst.exists():
        j = json.loads(src.read_text()); j["model"] = "yolo26m"; j["reused_from"] = src.name
        dst.write_text(json.dumps(j, indent=2))

for j0 in base:
    m, srcw = j0["model"], j0["source"]
    for s in SEEDS:
        name = f"{m}_s{s}"
        mf = MS / f"{name}.json"
        if mf.exists():
            continue
        run = SR / "train" / name
        pretrained = srcw.endswith(".pt")
        for attempt in range(3):
            try:
                t0 = time.time()
                last = run / "weights" / "last.pt"
                if (run / "TRAINED").exists():
                    pass
                elif last.exists():
                    log(f"{name}: resume"); YOLO(str(last)).train(resume=True)
                else:
                    log(f"{name}: egitim (pretrained={pretrained})")
                    YOLO(srcw).train(data=str(DATA), epochs=100, imgsz=640, batch=16, device=0, workers=2,
                                     cache="ram", seed=s, deterministic=True, pretrained=pretrained,
                                     project=str(SR / "train"), name=name, exist_ok=True, plots=False,
                                     save=True, val=True, verbose=False)
                (run / "TRAINED").touch()
                rc = pd.read_csv(run / "results.csv"); rc.columns = rc.columns.str.strip()
                train_h = float(rc["time"].iloc[-1]) / 3600 if "time" in rc else (time.time() - t0) / 3600
                best = YOLO(str(run / "weights" / "best.pt"))
                res = {"model": m, "source": srcw, "seed": s, "pretrained": pretrained, "epochs": 100,
                       "params_M": round(sum(p.numel() for p in best.model.parameters()) / 1e6, 3),
                       "GFLOPs": round(get_flops(best.model, 640) or 0.0, 2), "train_time_h": round(train_h, 3)}
                for split in ("val", "test"):
                    r = best.val(data=str(DATA), split=split, imgsz=640, batch=16, device=0,
                                 project=str(SR / f"eval_{split}"), name=name, exist_ok=True, plots=False,
                                 save_json=(split == "test"), verbose=False)
                    res[split] = metrics_of(r)
                mf.write_text(json.dumps(res, indent=2))
                if last.exists():
                    last.unlink()  # disk tasarrufu; best.pt korunur
                log(f"{name}: TAMAM val={res['val']['mAP50-95']:.4f} test={res['test']['mAP50-95']:.4f}")
                break
            except Exception as e:
                log(f"{name}: HATA (deneme {attempt + 1}) {e}")
                with open(SR / "errors.log", "a") as fh:
                    fh.write(f"\n===== {name} {time.ctime()} =====\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower():
                    time.sleep(300)
                else:
                    break
            finally:
                torch.cuda.empty_cache()
log("Tum tohumlar bitti.")
