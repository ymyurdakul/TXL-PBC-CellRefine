# TXL-PBC uzerinde Ultralytics YOLO modellerinin toplu egitimi ve degerlendirilmesi
# Calistirma: nohup python train_all_yolo.py > train_all.log 2>&1 &
# Yarida kalirsa ayni komutla tekrar calistirilir; biten modeller atlanir, yarim kalan resume edilir.
import os, sys, json, time, csv, traceback, platform
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # GPU0 baska bir islem tarafindan kullaniliyor
from pathlib import Path
import torch, ultralytics
from ultralytics import YOLO

ROOT = (Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1])) / "TXL-PBC_Dataset-master/TXL-PBC")
OUT = (Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1])) / "TXLPBC_runs")
EPOCHS = int(os.environ.get("EPOCHS", 100))
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
SEED = 0
NAMES = ["WBC", "RBC", "Platelets"]

# Kucukten buyuge; YOLOv6 icin resmi on-egitimli agirlik yok -> sifirdan (en sona)
MODELS = (
    ["yolov5nu.pt", "yolov8n.pt", "yolov9t.pt", "yolov10n.pt", "yolo11n.pt", "yolo12n.pt", "yolo26n.pt", "yolov3-tinyu.pt"]
  + ["yolov5su.pt", "yolov8s.pt", "yolov9s.pt", "yolov10s.pt", "yolo11s.pt", "yolo12s.pt", "yolo26s.pt"]
  + ["yolov5mu.pt", "yolov8m.pt", "yolov9m.pt", "yolov10m.pt", "yolo11m.pt", "yolo12m.pt", "yolo26m.pt"]
  + ["yolov10b.pt", "yolov9c.pt"]
  + ["yolov5lu.pt", "yolov8l.pt", "yolov10l.pt", "yolo11l.pt", "yolo12l.pt", "yolo26l.pt", "yolov3u.pt"]
  + ["yolov5xu.pt", "yolov8x.pt", "yolov9e.pt", "yolov10x.pt", "yolo11x.pt", "yolo12x.pt", "yolo26x.pt"]
  + ["yolov6n.yaml", "yolov6s.yaml", "yolov6m.yaml", "yolov6l.yaml", "yolov6x.yaml"]
)
if len(sys.argv) > 1:
    MODELS = [m for m in MODELS if Path(m).stem in sys.argv[1:]]

OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "txlpbc.yaml"
DATA.write_text(
    f"path: {ROOT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    f"nc: {len(NAMES)}\nnames: {NAMES}\n")
(OUT / "environment.json").write_text(json.dumps({
    "ultralytics": ultralytics.__version__, "torch": torch.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "python": platform.python_version(), "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
    "models": MODELS}, indent=2))

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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

def write_summary():
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

(OUT / "metrics").mkdir(exist_ok=True)
for m in MODELS:
    name = Path(m).stem
    run = OUT / "train" / name
    if (run / "DONE").exists():
        log(f"{name}: zaten tamamlanmis, atlaniyor"); continue
    pretrained = m.endswith(".pt")
    try:
        t0 = time.time()
        last = run / "weights" / "last.pt"
        if (run / "TRAINED").exists():
            log(f"{name}: egitim bitmis, sadece degerlendirme")
        elif last.exists():
            log(f"{name}: yarim kalan egitim resume ediliyor")
            YOLO(str(last)).train(resume=True)
        else:
            log(f"{name}: egitim basliyor (pretrained={pretrained})")
            YOLO(m).train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
                          workers=2, cache="ram", seed=SEED, deterministic=True, pretrained=pretrained,
                          project=str(OUT / "train"), name=name, exist_ok=True,
                          plots=True, save=True, val=True, verbose=True)
        train_h = (time.time() - t0) / 3600
        (run / "TRAINED").touch()

        best = YOLO(str(run / "weights" / "best.pt"))
        params = sum(x.numel() for x in best.model.parameters())
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(best.model, IMGSZ) or 0.0
        rcsv = run / 'results.csv'
        if train_h < 0.05 and rcsv.exists():
            import pandas as pd
            rc = pd.read_csv(rcsv); rc.columns = rc.columns.str.strip()
            train_h = float(rc['time'].iloc[-1]) / 3600 if 'time' in rc.columns else train_h
        res = {"model": name, "source": m, "pretrained": pretrained, "epochs": EPOCHS,
               "params_M": round(params / 1e6, 3), "GFLOPs": round(gflops, 2), "train_time_h": round(train_h, 3)}
        for split in ("val", "test"):
            r = best.val(data=str(DATA), split=split, imgsz=IMGSZ, batch=BATCH, device=0,
                         project=str(OUT / f"eval_{split}"), name=name, exist_ok=True,
                         plots=True, save_json=True, save_txt=True, save_conf=True)
            res[split] = metrics_of(r)
        best.predict(source=str(ROOT / "images" / "test"), imgsz=IMGSZ, device=0, conf=0.25,
                     project=str(OUT / "predict_test"), name=name, exist_ok=True,
                     save=True, save_txt=True, save_conf=True, verbose=False)
        (OUT / "metrics" / f"{name}.json").write_text(json.dumps(res, indent=2))
        write_summary()
        (run / "DONE").touch()
        log(f"{name}: TAMAM  test mAP50={res['test']['mAP50']:.4f}  mAP50-95={res['test']['mAP50-95']:.4f}")
    except Exception:
        log(f"{name}: HATA")
        traceback.print_exc()
        with open(OUT / "errors.log", "a") as fh:
            fh.write(f"\n===== {name} {time.ctime()} =====\n{traceback.format_exc()}")
    finally:
        torch.cuda.empty_cache()

write_summary()
log("Tum modeller bitti.")
