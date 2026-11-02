# Re-evaluates YOLO26-m (seed 0) on the TXL-PBC test set from predictions.json, replicating the Ultralytics
# validator: TP matching at IoU 0.50:0.95, 101-point AP, P/R/F1 curves, F1-optimal confidence, confusion matrix.
import json, glob, os, numpy as np, pandas as pd
R = 'results/benchmark'
PRED = 'TXLPBC_runs/eval_test/yolo26m/predictions.json'
LAB = 'TXL-PBC_Dataset-master/TXL-PBC/labels/test'
IMG = pd.read_csv('results/dataset/images.csv').set_index('img')
NAMES = ['WBC', 'RBC', 'Platelets']
IOUV = np.linspace(0.5, 0.95, 10)

def box_iou(a, b):  # a: (M,4) xyxy, b: (N,4) xyxy -> (M,N)
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)

preds = json.load(open(PRED))
by_img = {}
for d in preds:
    x, y, w, h = d['bbox']
    by_img.setdefault(d['image_id'], []).append([x, y, x + w, y + h, d['score'], d['category_id'] - 1])

gts = {}
for f in sorted(glob.glob(f'{LAB}/*.txt')):
    stem = os.path.basename(f)[:-4]; W, H = IMG.loc[stem + '.png', ['W', 'H']]
    rows = []
    for line in open(f):
        c, cx, cy, w, h = line.split()
        c = int(c); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, c])
    arr = np.array(rows).reshape(-1, 5)
    if len(arr):  # Ultralytics drops exact duplicate label rows when it verifies the labels
        _, keep = np.unique(np.round(arr, 6), axis=0, return_index=True); arr = arr[np.sort(keep)]
    gts[stem] = arr

def match(pred_cls, true_cls, iou):
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    matched = np.zeros((iou.shape[0], len(IOUV)), bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None]); k = available.argmax(0)
        correct[j] = available[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct

TP, CONF, PCLS, TCLS = [], [], [], []
for stem, g in gts.items():
    p = np.array(by_img.get(stem, [])).reshape(-1, 6)
    p = p[np.argsort(-p[:, 4], kind='stable')]
    TCLS.append(g[:, 4])
    if len(p) == 0: continue
    iou = box_iou(g[:, :4], p[:, :4]) if len(g) else np.zeros((0, len(p)))
    c = match(p[:, 5], g[:, 4], iou) if len(g) else np.zeros((len(p), 10), bool)
    TP.append(c); CONF.append(p[:, 4]); PCLS.append(p[:, 5])
tp = np.concatenate(TP); conf = np.concatenate(CONF); pcls = np.concatenate(PCLS); tcls = np.concatenate(TCLS)

# ---- ap_per_class replica
AP_X = np.linspace(0, 1, 101)
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return np.trapezoid(np.interp(AP_X, mrec, mpre), AP_X), mpre, mrec
def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1; p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0); return np.convolve(yp, np.ones(nf) / nf, mode='valid')
i = np.argsort(-conf); tp, conf, pcls = tp[i], conf[i], pcls[i]
uc, nt = np.unique(tcls, return_counts=True); nc = len(uc)
x = np.linspace(0, 1, 1000); ap = np.zeros((nc, 10)); pc = np.zeros((nc, 1000)); rc = np.zeros((nc, 1000)); prec_values = []
for ci, c in enumerate(uc):
    m = pcls == c; n_l = nt[ci]
    fpc = (1 - tp[m]).cumsum(0); tpc = tp[m].cumsum(0)
    recall = tpc / (n_l + 1e-16); precision = tpc / (tpc + fpc)
    rc[ci] = np.interp(-x, -conf[m], recall[:, 0], left=0); pc[ci] = np.interp(-x, -conf[m], precision[:, 0], left=1)
    for j in range(10):
        ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if j == 0: prec_values.append(np.interp(x, mrec, mpre))
prec_values = np.array(prec_values)
f1c = 2 * pc * rc / (pc + rc + 1e-16)
ib = smooth(f1c.mean(0), 0.1).argmax()
P, Rr, F1 = pc[:, ib], rc[:, ib], f1c[:, ib]
conf_best = x[ib]
res = dict(conf_best=float(conf_best), P=P.tolist(), R=Rr.tolist(), F1=F1.tolist(), AP50=ap[:, 0].tolist(), AP5095=ap.mean(1).tolist(),
           mP=float(P.mean()), mR=float(Rr.mean()), mAP50=float(ap[:, 0].mean()), mAP5095=float(ap.mean()), n_gt=nt.tolist())
print(json.dumps(res, indent=1))
ref = json.load(open('TXLPBC_runs/metrics/yolo26m.json'))['test']
print('reference:', {k: round(ref[k], 4) for k in ('P', 'R', 'mAP50', 'mAP50-95', 'AP50-95_Platelets', 'AP50_Platelets')})

# ---- confusion matrix replica (class-agnostic greedy matching, IoU 0.45)
def confusion(conf_thr=0.25, iou_thr=0.45):
    M = np.zeros((4, 4), int)  # rows predicted, cols true, index 3 = background
    for stem, g in gts.items():
        p = np.array(by_img.get(stem, [])).reshape(-1, 6); p = p[p[:, 4] > conf_thr]
        gc = g[:, 4].astype(int); dc = p[:, 5].astype(int)
        if len(g) == 0:
            for c in dc: M[c, 3] += 1
            continue
        if len(p) == 0:
            for c in gc: M[3, c] += 1
            continue
        iou = box_iou(g[:, :4], p[:, :4]); gt_match = np.full(len(g), -1)
        cand = np.flatnonzero((iou > iou_thr).any(0))
        for di in cand[np.argsort(-p[cand, 4], kind='stable')]:
            gi = iou[:, di].argmax()
            if iou[gi, di] > iou_thr:
                gt_match[gi] = di; iou[gi] = 0
        matched = set(gt_match[gt_match >= 0].tolist())
        for i_, c in enumerate(gc):
            di = gt_match[i_]
            if di >= 0: M[dc[di], c] += 1
            else: M[3, c] += 1
        for i_, c in enumerate(dc):
            if i_ not in matched: M[c, 3] += 1
    return M
cm25 = confusion(0.25); cmF1 = confusion(conf_best)
print('CM conf 0.25 (rows pred WBC,RBC,PLT,bg; cols true):\n', cm25)
print(f'CM conf {conf_best:.3f}:\n', cmF1)
# per-class recall-normalised
np.save(f'{R}/out/best_cm25.npy', cm25); np.save(f'{R}/out/best_cmF1.npy', cmF1)
np.savez(f'{R}/out/best_curves.npz', x=x, pc=pc, rc=rc, f1c=f1c, prec_values=prec_values, ap=ap, nt=nt, conf_best=conf_best, ib=ib)
json.dump(res, open(f'{R}/out/best_eval.json', 'w'), indent=1)
