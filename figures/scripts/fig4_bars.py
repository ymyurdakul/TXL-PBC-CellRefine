# Fig. 4 - Benchmark of the 45 detectors as bar charts: (a) test mAP50-95, (b) throughput.
import sys, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, 'figures/scripts')
from style import *
import matplotlib.pyplot as plt
from PIL import Image

S = pd.read_csv('results/benchmark/summary_test.csv').sort_values('rank').reset_index(drop=True)
R = pd.read_csv('results/benchmark/runs_long.csv')
BEST = 'yolo26m'
x = np.arange(len(S)); gaps = np.zeros(len(S))
fams = list(dict.fromkeys(S.family))
off = 0.0; xs = []
for i, f in enumerate(S.family):
    if i and f != S.family[i - 1]: off += 0.9
    xs.append(i + off)
xs = np.array(xs)
lab = {'base': 'base', 'tiny': 'tiny'}
ticks = [lab.get(v, v) for v in S.variant]
C_BAR, C_BEST = '#c9c8c3', INK2

W, H = 174.0, 85.0
fig = plt.figure(figsize=(W * MM, H * MM)); pg = Page(fig, W, H)
axa = pg.ax(14.0, 8.0, 158.0, 42.0); axb = pg.ax(14.0, 55.0, 158.0, 24.0)

# (a) mAP50-95
m = S['mAP50-95_mean'].values * 100; sd = S['mAP50-95_sd'].values * 100
YMIN = 84.0
cols = [C_BEST if mm == BEST else C_BAR for mm in S.model]
axa.bar(xs, m - YMIN, bottom=YMIN, width=0.72, color=cols, zorder=2, linewidth=0)
axa.errorbar(xs, m, yerr=sd, fmt='none', ecolor=INK, elinewidth=0.5, capsize=0.9, capthick=0.4, zorder=3)
col = [c for c in R.columns if c.lower().replace('_', '-') in ('map50-95', 'metrics/map50-95(b)')]
key = 'mAP50-95' if 'mAP50-95' in R.columns else col[0]
for i, mm in enumerate(S.model):
    v = R[(R.model == mm) & (R.split == 'test')][key].values if 'split' in R.columns else R[R.model == mm][key].values
    v = v * 100 if v.max() <= 1 else v
    axa.scatter(np.full(len(v), xs[i]), v, s=1.6, color=INK, zorder=4, linewidths=0)
axa.axhline(m[S.model == BEST][0], color=INK, lw=0.4, ls=(0, (2, 2)), zorder=1)
axa.set_ylim(YMIN, 89.0); axa.set_yticks(np.arange(84, 89.01, 1))
axa.set_ylabel('mAP50–95 (%)'); axa.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
axa.grid(axis='y', color=GRID, lw=0.4, zorder=0); axa.set_axisbelow(True)
axa.set_xticks(xs); axa.set_xticklabels([])
# axis break marks at the bottom of the y axis
for dy in (0.0, 0.18):
    axa.plot([-0.006, 0.006], [0.02 + dy * 0.12, 0.04 + dy * 0.12], transform=axa.transAxes, color=INK2, lw=0.6, clip_on=False)
axa.text(xs[S.model == BEST][0], m[S.model == BEST][0] + sd[S.model == BEST][0] + 0.12, 'YOLO26-m\n87.79', ha='center', va='bottom', fontsize=5.6, color=INK)
# family labels on top
for f in fams:
    xx = xs[S.family == f]
    axa.text(xx.mean(), 89.15, f, ha='center', va='bottom', fontsize=6.0, color=INK2)
    if f != fams[0]:
        for a in (axa, axb):
            a.axvline(xx[0] - 0.95, color=HAIR, lw=0.4, zorder=1)
pg.text(2.0, 5.0, 'a', fontsize=9, fontweight='bold')

# (b) throughput
fps = S['FPS_mean'].values
axb.bar(xs, fps, width=0.72, color=cols, zorder=2, linewidth=0)
axb.set_ylabel('Images per second'); axb.set_ylim(0, 450); axb.set_yticks([0, 100, 200, 300, 400])
axb.set_xlim(axa.get_xlim()); axb.grid(axis='y', color=GRID, lw=0.4, zorder=0); axb.set_axisbelow(True)
axb.set_xticks(xs); axb.set_xticklabels(ticks, fontsize=5.4, rotation=90)
axb.tick_params(axis='x', length=0, pad=1.5)
pg.text(2.0, 52.0, 'b', fontsize=9, fontweight='bold')

out = 'figures/output/Fig4_benchmark_bars'
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
Image.open(out + '.png').convert('RGB').save(out + '.tiff', compression='tiff_lzw', dpi=(600, 600))
print('ok', key)
