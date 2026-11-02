"""CellRefine ortak araclar: veri yukleme, dedektor tahmini, ultralytics ile ayni mAP hesabi."""
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TXLPBC_ROOT", Path(__file__).resolve().parents[1]))
DS = ROOT / "TXL-PBC_Dataset-master" / "TXL-PBC"
OUT = ROOT / "TXLPBC_runs" / "cellrefine"
NAMES = ["WBC", "RBC", "Platelets"]
IOUV = np.linspace(0.5, 0.95, 10)

# 3 tohumlu yolo26m dedektorleri (mevcut tablodaki temel modeller)
DETECTORS = {
    "yolo26m_s0": ROOT / "TXLPBC_runs/train/yolo26m/weights/best.pt",
    "yolo26m_s1": ROOT / "TXLPBC_runs/train_cell/base_yolo26m_s1/weights/best.pt",
    "yolo26m_s2": ROOT / "TXLPBC_runs/train_cell/base_yolo26m_s2/weights/best.pt",
}


def image_list(split):
    return sorted(p for p in glob.glob(str(DS / "images" / split / "*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")))


def label_path(img):
    p = Path(img)
    return str(p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt"))


def load_gt(img, wh):
    w, h = wh
    lp = label_path(img)
    if not os.path.exists(lp):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    a = np.loadtxt(lp, ndmin=2).astype(np.float32)
    if a.size:
        a = np.unique(a, axis=0)  # ultralytics gibi tekrarlanan etiketleri at
    if a.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    c, x, y, bw, bh = a[:, 0], a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    box = np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], 1)
    return box, c.astype(np.int64)


def predict(weights, split, imgsz=640, batch=16, device=0, tag=None):
    """Dedektor tahminleri, ultralytics val() ile ayni on-isleme (rect, half, conf=0.001, max_det=300).
    Cikti: {img_yolu: dict(box=(N,4) piksel xyxy, conf=(N,), cls=(N,), wh=(w,h))}"""
    import json

    from PIL import Image
    from ultralytics import YOLO

    m = YOLO(str(weights))
    tag = tag or Path(weights).parent.parent.name
    r = m.val(data=str(ROOT / "TXLPBC_runs" / "txlpbc.yaml"), split=split, imgsz=imgsz, batch=batch, device=device,
              save_json=True, plots=False, verbose=False, project=str(OUT / "val_json"), name=f"{tag}_{split}",
              exist_ok=True)
    js = json.loads((Path(r.save_dir) / "predictions.json").read_text())
    imgs = image_list(split)
    stem = {Path(p).stem: p for p in imgs}
    out = {}
    for p in imgs:
        with Image.open(p) as im:
            wh = im.size
        out[p] = dict(box=np.zeros((0, 4), np.float32), conf=np.zeros(0, np.float32), cls=np.zeros(0, np.int64), wh=wh)
    tmp = {}
    for d in js:
        tmp.setdefault(stem[str(d["image_id"])], []).append(d)
    for p, v in tmp.items():
        x, y, w, h = np.array([d["bbox"] for d in v], np.float32).T
        out[p]["box"] = np.stack([x, y, x + w, y + h], 1)
        out[p]["conf"] = np.array([d["score"] for d in v], np.float32)
        out[p]["cls"] = np.array([d["category_id"] - 1 for d in v], np.int64)
    return out, {"mAP50-95": float(r.box.map), "mAP50": float(r.box.map50)}


def box_iou(a, b):
    """a (N,4), b (M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    bb = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-7)


def match_predictions(pred_cls, true_cls, iou):
    """ultralytics BaseValidator.match_predictions ile ayni (iou: gt x pred)."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def evaluate(preds, gts):
    """preds/gts: {img: dict(box, conf, cls)} / {img: (box, cls)} -> metrik sozlugu (ultralytics ap_per_class)."""
    from ultralytics.utils.metrics import ap_per_class

    tps, confs, pcls, tcls = [], [], [], []
    for k, (gb, gc) in gts.items():
        p = preds.get(k)
        tcls.append(gc)
        if p is None or len(p["box"]) == 0:
            continue
        iou = box_iou(gb, p["box"])
        tps.append(match_predictions(p["cls"], gc, iou))
        confs.append(p["conf"])
        pcls.append(p["cls"])
    tp = np.concatenate(tps, 0)
    conf = np.concatenate(confs)
    pc = np.concatenate(pcls)
    tc = np.concatenate(tcls)
    r = ap_per_class(tp, conf, pc, tc)
    p, rc, ap, uc = r[2], r[3], r[5], r[6]
    d = {"P": float(p.mean()), "R": float(rc.mean()), "mAP50": float(ap[:, 0].mean()), "mAP75": float(ap[:, 5].mean()),
         "mAP50-95": float(ap.mean())}
    for j, c in enumerate(uc):
        d[f"AP50-95_{NAMES[int(c)]}"] = float(ap[j].mean())
    return d


def gt_dict(preds_or_split):
    if isinstance(preds_or_split, str):
        from PIL import Image
        g = {}
        for p in image_list(preds_or_split):
            with Image.open(p) as im:
                g[p] = load_gt(p, im.size)
        return g
    return {k: load_gt(k, v["wh"]) for k, v in preds_or_split.items()}


def cached_preds(det, split):
    import pickle

    f = OUT / "preds" / f"{det}_v_{split}.pkl"
    if f.exists():
        # cached predictions are keyed by the image paths of the original machine; map them to this dataset location
        return {str(DS / "images" / split / Path(k).name): v for k, v in pickle.loads(f.read_bytes()).items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    w = DETECTORS.get(det)
    if w is None:  # ornek: yolo11m (tohum 0) veya yolo11m_s1 (seeds/train)
        w = ROOT / "TXLPBC_runs" / ("seeds/train" if det[-3:-1] == "_s" else "train") / det / "weights" / "best.pt"
    p, _ = predict(w, split, tag=det)
    f.write_bytes(pickle.dumps(p))
    return p
