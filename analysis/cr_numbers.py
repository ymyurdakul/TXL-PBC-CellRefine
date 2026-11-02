# All CellRefine numbers used in figures, tables and text (from the raw metrics JSON, hist.json and logs).
import json, re, glob, numpy as np, pandas as pd
from scipy import stats
D = pd.read_pickle('analysis/cr_all.pkl')
R = 'TXLPBC_runs/cellrefine'
S = pd.read_csv('results/benchmark/summary_test.csv', index_col=0)
RL = pd.read_csv('results/benchmark/runs_long.csv')
K = ['P', 'R', 'mAP50', 'mAP75', 'mAP50-95', 'AP50-95_WBC', 'AP50-95_RBC', 'AP50-95_Platelets']
out = {}
def grp(files):
    g = {}
    for sp in ('val', 'test'):
        for k in K:
            b = np.array([D.loc[f, f'base_{sp}.{k}'] for f in files]) * 100; r = np.array([D.loc[f, f'{sp}.{k}'] for f in files]) * 100
            g[f'{sp}.{k}'] = dict(base=b.tolist(), ref=r.tolist(), d=(r - b).tolist())
    g['ms'] = [float(D.loc[f, 'ms']) for f in files]; g['boxes'] = [float(D.loc[f, 'boxes']) for f in files]
    g['chosen'] = [(int(D.loc[f, 'iters']), float(D.loc[f, 'lam'])) for f in files]; g['params'] = float(D.loc[files[0], 'params'])
    return g
out['main'] = grp(['r18_s0_lam', 'r18_s1_lam', 'r18_s2_lam'])
out['light'] = grp(['l3_12s_s0_lam', 'l3_12s_s1_lam', 'l3_12s_s2_lam'])
# detector cost (seed means from the benchmark) and throughput
for det, key in (('yolo26m', 'main'), ('yolo12s', 'light')):
    r = S.loc[det]; out[key]['det_params'] = float(r.params_M); out[key]['det_gflops'] = float(r.GFLOPs)
    t = RL[(RL.model == det) & (RL.split == 'test')]
    out[key]['det_ms'] = t.total_ms.tolist()
# plug-in: refiner r18_s0 (trained on YOLO26-m seed 0 predictions) on 12 detectors (seed 0)
plug = {'yolo26m': 'r18_s0_lam'}
for f in D.index:
    m = re.match(r'r18_s0_on_(.+)$', f)
    if m: plug[m.group(1)] = f
P = {}
for det, f in plug.items():
    P[det] = {k: dict(base=float(D.loc[f, f'base_test.{k}'] * 100), ref=float(D.loc[f, f'test.{k}'] * 100)) for k in K}
    P[det]['val'] = dict(base=float(D.loc[f, 'base_val.mAP50-95'] * 100), ref=float(D.loc[f, 'val.mAP50-95'] * 100))
    P[det]['chosen'] = (int(D.loc[f, 'iters']), float(D.loc[f, 'lam'])); P[det]['ms'] = float(D.loc[f, 'ms']); P[det]['boxes'] = float(D.loc[f, 'boxes'])
    P[det]['params'] = float(S.loc[det, 'params_M']); P[det]['gflops'] = float(S.loc[det, 'GFLOPs'])
out['plug'] = P
d = np.array([P[k]['mAP50-95']['ref'] - P[k]['mAP50-95']['base'] for k in P])
out['plug_stats'] = dict(n=len(d), min=float(d.min()), max=float(d.max()), mean=float(d.mean()), median=float(np.median(d)),
                         wilcoxon_p=float(stats.wilcoxon(d, alternative='two-sided').pvalue))
# other refiner variants on YOLO26-m seed 0 and on YOLO12-s seed 0
V = {}
for f in ['r18_s0_lam', 'l3_s0_lam', 'l3kd_s0_lam', 'r18_s0_tta8', 'l3_s0_on_yolo12s', 'l3kd_s0_on_yolo12s', 'r18_s0_on_yolo12s', 'l3_12s_s0_lam', 'l3kd_s0_on_yolo12n', 'r18_s0_on_yolo12n']:
    V[f] = dict(det=D.loc[f, 'det'], params=float(D.loc[f, 'params']), val_base=float(D.loc[f, 'base_val.mAP50-95'] * 100), val=float(D.loc[f, 'val.mAP50-95'] * 100),
                test_base=float(D.loc[f, 'base_test.mAP50-95'] * 100), test=float(D.loc[f, 'test.mAP50-95'] * 100),
                plt_base=float(D.loc[f, 'base_test.AP50-95_Platelets'] * 100), plt=float(D.loc[f, 'test.AP50-95_Platelets'] * 100),
                chosen=(int(D.loc[f, 'iters']), float(D.loc[f, 'lam'])), ms=float(D.loc[f, 'ms']), tta=int(D.loc[f, 'tta']))
out['variants'] = V
# inference grid of the first evaluation of r18_s0 (iterations 0-2, beta 0-0.5, lambda 1)
out['grid_r18_s0'] = {k: v * 100 for k, v in D.loc['r18_s0', 'grid'].items()}
out['grid_main'] = {f: {k: v * 100 for k, v in D.loc[f, 'grid'].items()} for f in ['r18_s0_lam', 'r18_s1_lam', 'r18_s2_lam']}
out['grid_tta8'] = {k: v * 100 for k, v in D.loc['r18_s0_tta8', 'grid'].items()}
# training histories (validation every 4 or 10 epochs) and design runs
H = {}
for f in glob.glob(f'{R}/runs/*/hist.json'):
    n = f.split('/')[-2]; j = json.load(open(f)); a = json.load(open(f.replace('hist.json', 'args.json')))
    H[n] = dict(base=j['base_val']['mAP50-95'] * 100, ep=[h['ep'] for h in j['hist']], ref=[h['i1_b0.0'] * 100 for h in j['hist']],
                resc=[h['i0_b0.5'] * 100 for h in j['hist']], both=[h['i1_b0.5'] * 100 for h in j['hist']],
                best_ref=max(h['i1_b0.0'] for h in j['hist']) * 100, args=a)
out['hist'] = H
# training time from logs
T = {}
for f in glob.glob(f'{R}/logs/*.log'):
    n = f.split('/')[-1][:-4]
    ls = [l for l in open(f, errors='ignore').read().replace('\r', '\n').split('\n') if l.startswith('ep ')]
    if ls:
        m = re.search(r'\| (\d+)s', ls[-1]); e = re.match(r'ep (\d+)/(\d+)', ls[-1])
        T[n] = dict(sec=int(m.group(1)), ep=int(e.group(1)), of=int(e.group(2)))
out['train_time'] = T
# box voting (YOLO26-m seed 0)
V2 = {}
for l in open(f'{R}/logs/vote_s0.log', errors='ignore').read().replace('\r', '\n').split('\n'):
    m = re.match(r'yolo26m_s0 (val|test) (.+?): mAP50-95=([\d.]+)', l)
    if m: V2[f'{m.group(1)}|{m.group(2)}'] = float(m.group(3)) * 100
out['voting'] = V2
json.dump(out, open('results/analysis/cr_numbers.json', 'w'), indent=1, default=float)
# summary print
for key in ('main', 'light'):
    g = out[key]
    for k in ('test.mAP50-95', 'test.mAP75', 'test.mAP50', 'test.AP50-95_Platelets', 'val.mAP50-95'):
        b, r, dd = map(np.array, (g[k]['base'], g[k]['ref'], g[k]['d']))
        print(f"{key:5s} {k:24s} base {b.mean():.2f}±{b.std(ddof=1):.2f} ref {r.mean():.2f}±{r.std(ddof=1):.2f} d {dd.mean():+.2f}±{dd.std(ddof=1):.2f}")
print('plug', out['plug_stats'])
for k, v in sorted(P.items(), key=lambda x: x[1]['mAP50-95']['base']):
    print(f"{k:10s} {v['mAP50-95']['base']:.2f} -> {v['mAP50-95']['ref']:.2f} ({v['mAP50-95']['ref']-v['mAP50-95']['base']:+.2f}) PLT {v['AP50-95_Platelets']['ref']-v['AP50-95_Platelets']['base']:+.2f} mAP75 {v['mAP75']['ref']-v['mAP75']['base']:+.2f} val {v['val']['ref']-v['val']['base']:+.2f} {v['chosen']} {v['ms']}")
for n, h in sorted(H.items()): print(n, round(h['base'], 2), 'best refined', round(h['best_ref'], 2), 'ep', h['ep'][int(np.argmax(h['ref']))], 'max rescore', round(max(h['resc']), 2))
print(T); print(V2)
