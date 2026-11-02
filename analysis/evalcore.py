# Replica of the Ultralytics 8.4.191 detection metrics (greedy matching at IoU 0.50:0.95, 101-point AP).
import numpy as np, glob, os, pickle, pandas as pd
IOUV = np.linspace(0.5, 0.95, 10)
LAB = 'TXL-PBC_Dataset-master/TXL-PBC/labels'
IMG = pd.read_csv('results/dataset/images.csv').set_index('img')
PRED = 'TXLPBC_runs/cellrefine/preds'

def box_iou(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-7)

def load_gt(split):
    G = {}
    for f in sorted(glob.glob(f'{LAB}/{split}/*.txt')):
        stem = os.path.basename(f)[:-4]; W, H = IMG.loc[stem + '.png', ['W', 'H']]
        a = np.loadtxt(f, ndmin=2).astype(np.float64)
        if a.size: a = a[np.sort(np.unique(np.round(a, 6), axis=0, return_index=True)[1])]
        c, x, y, w, h = a[:, 0], a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
        G[stem] = (np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1), c.astype(int))
    return G

def load_pred(det, split):
    P = pickle.load(open(f'{PRED}/{det}_v_{split}.pkl', 'rb'))
    return {os.path.basename(k)[:-4]: v for k, v in P.items()}

def match(pred_cls, true_cls, iou):
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    matched = np.zeros((iou.shape[0], len(IOUV)), bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None]); k = available.argmax(0)
        correct[j] = available[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct

AP_X = np.linspace(0, 1, 101)
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return np.trapezoid(np.interp(AP_X, mrec, mpre), AP_X)

def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1; p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0); return np.convolve(yp, np.ones(nf) / nf, mode='valid')

def evaluate(P, G):
    """P: {stem: dict(box, conf, cls)}, G: {stem: (box, cls)} -> dict with ap (3x10), mAP50, mAP75, mAP50-95, P, R."""
    TP, CF, PC, TC = [], [], [], []
    for stem, (gb, gc) in G.items():
        TC.append(gc); p = P.get(stem)
        if p is None or len(p['box']) == 0: continue
        o = np.argsort(-p['conf'], kind='stable')
        b, c, s = p['box'][o], p['cls'][o], p['conf'][o]
        TP.append(match(c, gc, box_iou(gb, b)) if len(gb) else np.zeros((len(b), 10), bool)); CF.append(s); PC.append(c)
    tp = np.concatenate(TP); conf = np.concatenate(CF); pcls = np.concatenate(PC); tcls = np.concatenate(TC)
    i = np.argsort(-conf); tp, conf, pcls = tp[i], conf[i], pcls[i]
    uc, nt = np.unique(tcls, return_counts=True)
    x = np.linspace(0, 1, 1000); ap = np.zeros((len(uc), 10)); pc = np.zeros((len(uc), 1000)); rc = np.zeros((len(uc), 1000))
    for ci, c in enumerate(uc):
        m = pcls == c; fpc = (1 - tp[m]).cumsum(0); tpc = tp[m].cumsum(0)
        recall = tpc / (nt[ci] + 1e-16); precision = tpc / (tpc + fpc)
        rc[ci] = np.interp(-x, -conf[m], recall[:, 0], left=0); pc[ci] = np.interp(-x, -conf[m], precision[:, 0], left=1)
        for j in range(10): ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * pc * rc / (pc + rc + 1e-16); ib = smooth(f1.mean(0), 0.1).argmax()
    return dict(ap=ap, mAP50=ap[:, 0].mean(), mAP75=ap[:, 5].mean(), map=ap.mean(), P=pc[:, ib].mean(), R=rc[:, ib].mean(),
                cls_map=ap.mean(1))
