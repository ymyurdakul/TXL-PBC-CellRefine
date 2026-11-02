# Per-detection comparison of YOLO26-m boxes before and after CellRefine (test split, 3 seeds).
import sys, os, pickle, json, numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
import evalcore as E
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
IM = pd.read_csv('results/dataset/images.csv').set_index('img')
NAMES = ['WBC', 'RBC', 'Platelets']
CONF = 0.25
G = E.load_gt('test')
tos = lambda P: {os.path.basename(k)[:-4]: v for k, v in P.items()}

def edges_err(b, g):
    w, h = g[2] - g[0], g[3] - g[1]
    return np.abs(b - g) / np.array([w, h, w, h])

rows = []
for s in (0, 1, 2):
    B, Rf = tos(R[(s, 'test')]['base']), tos(R[(s, 'test')]['ref'])
    for stem, (gb, gc) in G.items():
        W, H = IM.loc[stem + '.png', ['W', 'H']]; src = IM.loc[stem + '.png', 'src']
        p, q = B[stem], Rf[stem]
        keep = p['conf'] >= CONF
        pb, qb, pc, pconf = p['box'][keep], q['box'][keep], p['cls'][keep], p['conf'][keep]
        gg = E.box_iou(gb, gb); np.fill_diagonal(gg, 0)
        iou = E.box_iou(gb, pb) * (gc[:, None] == pc[None, :])
        used_g, used_p = set(), set()
        for gi, pi in sorted(zip(*np.nonzero(iou >= 0.5)), key=lambda t: -iou[t[0], t[1]]):
            if gi in used_g or pi in used_p: continue
            used_g.add(gi); used_p.add(pi)
            i0 = iou[gi, pi]; i1 = E.box_iou(gb[gi:gi + 1], qb[pi:pi + 1])[0, 0]
            g = gb[gi]
            e0, e1 = edges_err(pb[pi], g), edges_err(qb[pi], g)
            border = (g[0] <= 2) or (g[1] <= 2) or (g[2] >= W - 2) or (g[3] >= H - 2)
            rows.append(dict(seed=s, img=stem, src=src, cls=NAMES[gc[gi]], gi=gi, pi=pi, conf=pconf[pi], iou0=i0, iou1=i1,
                             e0=e0.mean(), e1=e1.mean(), border=border, ovl=gg[gi].max() if len(gb) > 1 else 0.0,
                             size=np.sqrt((g[2] - g[0]) * (g[3] - g[1]))))
        rows_fp = None
df = pd.DataFrame(rows); df['d'] = df.iou1 - df.iou0
df.to_csv('results/analysis/pairs.csv', index=False)
T = 0.01
def summ(x):
    return pd.Series(dict(n=len(x) / 3, iou0=x.iou0.mean() * 100, iou1=x.iou1.mean() * 100, d=x.d.mean() * 100,
                          imp=(x.d > T).mean() * 100, wor=(x.d < -T).mean() * 100, same=(x.d.abs() <= T).mean() * 100,
                          e0=x.e0.mean() * 100, e1=x.e1.mean() * 100,
                          h90_0=(x.iou0 >= .9).mean() * 100, h90_1=(x.iou1 >= .9).mean() * 100,
                          h95_0=(x.iou0 >= .95).mean() * 100, h95_1=(x.iou1 >= .95).mean() * 100,
                          h75_0=(x.iou0 >= .75).mean() * 100, h75_1=(x.iou1 >= .75).mean() * 100))
pd.set_option('display.width', 250); pd.set_option('display.precision', 2)
print('ALL'); print(summ(df).to_frame().T)
print(df.groupby('cls').apply(summ))
print(df.groupby('src').apply(summ))
print(df.groupby('border').apply(summ))
df['ovlb'] = pd.cut(df.ovl, [-1, 0.0001, 0.1, 1], labels=['none', '<0.1', '>=0.1'])
print(df.groupby('ovlb').apply(summ))
df['bin'] = pd.cut(df.iou0, [0.5, 0.7, 0.8, 0.9, 0.95, 1.0], right=False)
print(df.groupby('bin').apply(summ))
print(df.groupby(['cls', 'bin']).apply(summ)[['n', 'd', 'imp', 'wor']])
# per-seed spread of key quantities
ps = df.groupby(['seed', 'cls']).apply(lambda x: pd.Series(dict(d=x.d.mean() * 100, imp=(x.d > T).mean() * 100, wor=(x.d < -T).mean() * 100)))
print(ps.groupby('cls').agg(['mean', 'std']))
ps2 = df.groupby('seed').apply(lambda x: pd.Series(dict(d=x.d.mean() * 100, imp=(x.d > T).mean() * 100, wor=(x.d < -T).mean() * 100, i0=x.iou0.mean()*100, i1=x.iou1.mean()*100)))
print(ps2, ps2.mean(), ps2.std())
