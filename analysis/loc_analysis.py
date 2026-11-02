# Localisation analysis of the detector stage from the cached detector predictions (no refiner weights needed).
import sys, json, numpy as np
sys.path.insert(0, 'analysis')
from evalcore import *
NAMES = ['WBC', 'RBC', 'Platelets']

def oracle(P, G, mode):
    out = {}
    for k, p in P.items():
        q = {kk: (vv.copy() if hasattr(vv, 'copy') else vv) for kk, vv in p.items()}
        gb, gc = G[k]
        if len(p['box']) and len(gb):
            iou = box_iou(p['box'], gb) * (p['cls'][:, None] == gc[None, :]); j = iou.argmax(1); best = iou.max(1)
            if mode == 'score': q['conf'] = best.astype(np.float32)
            elif mode == 'loc':
                m = best >= 0.5; q['box'][m] = gb[j[m]]
        out[k] = q
    return out

def matched_stats(P, G, conf=0.25):
    I, E, C = [], [], []
    for k, p in P.items():
        gb, gc = G[k]
        if not len(p['box']) or not len(gb): continue
        iou = box_iou(p['box'], gb) * (p['cls'][:, None] == gc[None, :]); j = iou.argmax(1); best = iou.max(1)
        sel = (best >= 0.5) & (p['conf'] >= conf)
        g = gb[j[sel]]; wh = np.maximum(g[:, 2:] - g[:, :2], 1)
        E.append((p['box'][sel] - g) / np.concatenate([wh, wh], 1)); I.append(best[sel]); C.append(p['cls'][sel])
    return np.concatenate(I), np.concatenate(E), np.concatenate(C)

res = {}
for sp in ('val', 'test'):
    G = load_gt(sp)
    for s in (0, 1, 2):
        det = f'yolo26m_s{s}'; P = load_pred(det, sp)
        r = {m: evaluate(oracle(P, G, m) if m != 'base' else P, G) for m in ('base', 'score', 'loc')}
        I, E, C = matched_stats(P, G)
        res[f'{sp}_{s}'] = dict(
            base_ap=r['base']['ap'].tolist(), base=r['base']['map'], score=r['score']['map'], loc=r['loc']['map'],
            score_cls=r['score']['cls_map'].tolist(), loc_cls=r['loc']['cls_map'].tolist(), base_cls=r['base']['cls_map'].tolist(),
            iou=[I[C == c].tolist() for c in range(3)], err=[E[C == c].tolist() for c in range(3)])
        print(sp, s, 'base %.4f score-oracle %.4f loc-oracle %.4f' % (r['base']['map'], r['score']['map'], r['loc']['map']),
              '| cls base', np.round(r['base']['cls_map'] * 100, 2), 'loc', np.round(r['loc']['cls_map'] * 100, 2))
        for c in range(3):
            i = I[C == c]; e = np.abs(E[C == c])
            print('   %-9s n=%4d meanIoU %.4f median %.4f IoU<0.75 %.3f IoU<0.85 %.3f  |edge err| mean %.4f median %.4f' % (
                NAMES[c], len(i), i.mean(), np.median(i), (i < 0.75).mean(), (i < 0.85).mean(), e.mean(), np.median(e)))
json.dump(res, open('loc_analysis.json', 'w'))
