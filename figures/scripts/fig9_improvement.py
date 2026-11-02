# Fig. 9 - Where CellRefine changes the YOLO26-m results (test set, three runs):
# (a) AP gain at each IoU threshold per class; (b) share of matched detections whose IoU improved, stayed, or worsened.
import sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, 'figures/scripts')
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

A = pickle.load(open('results/analysis/ap_thr.pkl', 'rb')); A0, A1 = A['A0'], A['A1']   # (seed, class, thr) in %
D = A1 - A0
df = pd.read_csv('results/analysis/pairs.csv'); df['d'] = df.iou1 - df.iou0
TH = 0.01
NM = {'WBC': 'White blood cell', 'RBC': 'Red blood cell', 'Platelets': 'Platelet'}

W, H = 174.0, 98.0
fig = plt.figure(figsize=(W * MM, H * MM)); pg = Page(fig, W, H)

# (a) AP gain per IoU threshold
axa = pg.ax(12.0, 9.0, 62.0, 70.0)
thr = np.linspace(0.5, 0.95, 10); xx = np.arange(10)
bw = 0.26
for k, c in enumerate(CLS):
    axa.bar(xx + (k - 1) * bw, D[:, k].mean(0), width=bw, color=COL[c], label=NM[c], zorder=2, linewidth=0)
mm = D.mean(1)                                     # mAP difference per seed
axa.errorbar(xx, mm.mean(0), yerr=mm.std(0, ddof=1), fmt='o', ms=2.6, color=INK, elinewidth=0.6, capsize=1.2, capthick=0.5,
             zorder=4, label='All classes (mean ± SD)')
axa.axhline(0, color=INK2, lw=0.5, zorder=3)
axa.set_xticks(xx); axa.set_xticklabels([f'{t:.2f}' for t in thr], rotation=90)
axa.set_xlabel('IoU threshold'); axa.set_ylabel('AP gain with CellRefine (points)')
axa.set_ylim(-1.0, 4.5); axa.grid(axis='y', color=GRID, lw=0.4, zorder=0); axa.set_axisbelow(True)
axa.legend(loc='upper left', fontsize=5.8, handlelength=1.2, borderaxespad=0.2)
pg.text(1.5, 5.0, 'a', fontsize=9, fontweight='bold')

# (b) improved / unchanged / worsened
rows = []
def add(label, sub, group):
    n = len(sub) / 3
    rows.append(dict(label=label, group=group, n=n, imp=(sub.d > TH).mean() * 100, same=(sub.d.abs() <= TH).mean() * 100,
                     wor=(sub.d < -TH).mean() * 100, md=sub.d.mean() * 100))
for lo, hi, lab in ((0.5, 0.7, '0.50–0.70'), (0.7, 0.8, '0.70–0.80'), (0.8, 0.9, '0.80–0.90'), (0.9, 0.95, '0.90–0.95'), (0.95, 1.01, '≥ 0.95')):
    add(lab, df[(df.iou0 >= lo) & (df.iou0 < hi)], 'IoU before refinement')
for c in CLS:
    add(NM[c], df[df.cls == c], 'Class')
add('No overlap', df[df.ovl <= 0], 'Overlap with a neighbor')
add('IoU < 0.1', df[(df.ovl > 0) & (df.ovl < 0.1)], 'Overlap with a neighbor')
add('IoU ≥ 0.1', df[df.ovl >= 0.1], 'Overlap with a neighbor')
for s in SRC:
    add(s, df[df.src == s], 'Source')
add('Not touching', df[~df.border], 'Image border')
add('Touching', df[df.border], 'Image border')
add('All matched detections', df, 'All')
Rb = pd.DataFrame(rows)
Rb.to_csv('results/analysis/fig9b_values.csv', index=False)

axb = pg.ax(108.0, 9.0, 48.0, 76.0)
y = 0; ys = []; gl = []; prev = None
for i, r in Rb.iterrows():
    if r.group != prev:
        if prev is not None: y += 0.75
        gl.append((y, r.group)); y += 1.0; prev = r.group
    ys.append(y); y += 1.0
ys = np.array(ys)
C_IMP, C_SAME, C_WOR = INK, '#e7e6e2', '#a9a8a3'
axb.barh(ys, Rb.imp, height=0.72, color=C_IMP, zorder=2, linewidth=0)
axb.barh(ys, Rb.same, left=Rb.imp, height=0.72, color=C_SAME, zorder=2, linewidth=0)
axb.barh(ys, Rb.wor, left=Rb.imp + Rb.same, height=0.72, color=C_WOR, zorder=2, linewidth=0)
for yy, r in zip(ys, Rb.itertuples()):
    axb.text(-1.5, yy, r.label, ha='right', va='center', fontsize=5.8, color=INK, fontweight='bold' if r.group == 'All' else 'normal')
    if r.imp >= 7: axb.text(r.imp / 2, yy, f'{r.imp:.0f}', ha='center', va='center', fontsize=5.0, color='white')
    if r.wor >= 7: axb.text(100 - r.wor / 2, yy, f'{r.wor:.0f}', ha='center', va='center', fontsize=5.0, color=INK)
    md = 0.0 if abs(r.md) < 0.005 else r.md
    axb.text(103, yy, (f'{md:+.2f}'.replace('-', '−')) if md else '0.00', ha='left', va='center', fontsize=5.6, color=INK)
    axb.text(119, yy, f'{r.n:.0f}', ha='left', va='center', fontsize=5.6, color=INK2)
for yy, g in gl:
    if g == 'All': continue
    axb.text(-40, yy, g, ha='left', va='center', fontsize=6.0, color=INK2, fontweight='bold')
axb.text(103, -1.3, 'Mean', ha='left', va='center', fontsize=5.6, color=INK2)
axb.text(103, -0.3, 'ΔIoU', ha='left', va='center', fontsize=5.6, color=INK2)
axb.text(119, -0.3, 'n', ha='left', va='center', fontsize=5.6, color=INK2)
axb.set_ylim(ys[-1] + 0.8, -1.9); axb.set_xlim(0, 100)
axb.set_yticks([]); axb.spines['left'].set_visible(False)
axb.set_xticks([0, 25, 50, 75, 100]); axb.set_xlabel('Matched detections (%)')
leg = [Patch(color=C_IMP, label='IoU improved'), Patch(color=C_SAME, label='Unchanged (within ±1 point)'), Patch(color=C_WOR, label='IoU worsened')]
fig.legend(handles=leg, loc='center', bbox_to_anchor=((108 + 24) / W, 1 - 95.0 / H), ncol=3, fontsize=5.8, handlelength=1.1,
           columnspacing=1.0, frameon=False)
pg.text(76.0, 5.0, 'b', fontsize=9, fontweight='bold')

out = 'figures/output/Fig9_improvement_analysis'
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
Image.open(out + '.png').convert('RGB').save(out + '.tiff', compression='tiff_lzw', dpi=(600, 600))
print(Rb.round(2).to_string())
