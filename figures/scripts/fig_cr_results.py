# Figures 7-10: localisation analysis of the detector stage and CellRefine results.
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
sys.path.insert(0, 'figures/scripts'); sys.path.insert(0, 'analysis')
from style import COL, INK, INK2, MUTED, GRID, HAIR, MM
from evalcore import load_gt, load_pred, box_iou, IMG
plt.rcParams.update({'font.size': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic'})
OUT = 'figures/output'
CLS = ['WBC', 'RBC', 'Platelets']; CLSNAME = {'WBC': 'White blood cell', 'RBC': 'Red blood cell', 'Platelets': 'Platelet'}
BB, NK, HD = '#5B3FA0', '#A35F00', '#A3245F'
GREY2 = '#6e6e6a'; BASE = '#a9a8a3'
N = json.load(open('results/analysis/cr_numbers.json'))
L = json.load(open('results/analysis/loc_analysis.json'))
S = pd.read_csv('results/benchmark/summary_test.csv', index_col=0).sort_values('rank')
LABEL = {m: (f if v == 'base' else f'{f}-{v}') for m, f, v in zip(S.index, S.family, S.variant)}
FCOL = {'YOLOv3': '#9a8c98', 'YOLOv5': '#6d8fa6', 'YOLOv6': '#b08968', 'YOLOv8': '#7a9e7e', 'YOLOv9': '#8d7bb0',
        'YOLOv10': '#c38d9e', 'YOLO11': '#5f7f9a', 'YOLO12': '#a67c52', 'YOLO26': '#3a3a3a', 'RT-DETR': '#b5651d'}
FMK = {'YOLOv3': 'v', 'YOLOv5': 's', 'YOLOv6': 'P', 'YOLOv8': 'D', 'YOLOv9': '^', 'YOLOv10': 'X', 'YOLO11': 'o',
       'YOLO12': 'h', 'YOLO26': '*', 'RT-DETR': 'd'}

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=600); fig.savefig(f'{OUT}/{name}.pdf')
    Image.open(f'{OUT}/{name}.png').convert('RGB').save(f'{OUT}/{name}.tiff', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig); print('saved', name)

def panel_label(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', ha='right', va='bottom', color=INK)

def tidy(ax, grid='y'):
    ax.grid(axis=grid, color=GRID, lw=0.4); ax.set_axisbelow(True)

# ===================================================================== Fig. 7: localisation analysis (detector stage)
fig = plt.figure(figsize=(174 * MM, 104 * MM))
IOU_T = np.linspace(0.5, 0.95, 10)
# a: AP at each IoU threshold
ax = fig.add_axes([0.075, 0.555, 0.25, 0.385])
for ci, c in enumerate(CLS):
    A = np.array([L[f'test_{s}']['base_ap'][ci] for s in range(3)]) * 100
    ax.fill_between(IOU_T, A.min(0), A.max(0), color=COL[c], alpha=0.18, lw=0)
    ax.plot(IOU_T, A.mean(0), color=COL[c], lw=1.0, marker='o', ms=2.2, label=CLSNAME[c])
ax.set_xlabel('IoU threshold'); ax.set_ylabel('AP (%)'); ax.set_xlim(0.48, 0.97); ax.set_ylim(0, 103)
ax.set_xticks(IOU_T[::2]); tidy(ax, 'both'); ax.legend(loc='lower left', handlelength=1.4)
ax.set_title('AP of YOLO26-m at each IoU threshold', fontsize=6.5, loc='left', color=INK)
panel_label(ax, 'a', x=-0.17)
# b: distribution of IoU of matched detections
ax = fig.add_axes([0.405, 0.555, 0.25, 0.385])
grid = np.linspace(0.5, 1.0, 201)
for ci, c in enumerate(CLS):
    v = np.concatenate([L[f'test_{s}']['iou'][ci] for s in range(3)])
    ax.plot(grid, [(v < g).mean() * 100 for g in grid], color=COL[c], lw=1.0, label=CLSNAME[c])
for t in (0.75, 0.85, 0.95):
    ax.axvline(t, color=HAIR, lw=0.4, ls=(0, (2, 2)), zorder=0)
ax.set_xlabel('IoU with the annotation'); ax.set_ylabel('Detections below this IoU (%)'); ax.set_xlim(0.6, 1.0); ax.set_ylim(0, 100)
tidy(ax, 'both'); ax.set_title('IoU of correct detections (score ≥ 0.25)', fontsize=6.5, loc='left', color=INK)
panel_label(ax, 'b', x=-0.17)
# c: oracle analysis
ax = fig.add_axes([0.735, 0.555, 0.25, 0.385])
grpn = CLS + ['mean']; xs = np.arange(4); wbar = 0.26
modes = [('base', 'Detector', BASE), ('score', 'Perfect scores', GREY2), ('loc', 'Perfect boxes', INK)]
for k, (m, lab, col) in enumerate(modes):
    vals = []
    for s in range(3):
        d = L[f'test_{s}']
        vals.append([v * 100 for v in d[f'{m}_cls']] + [d[m] * 100])
    vals = np.array(vals)
    ax.bar(xs + (k - 1) * wbar, vals.mean(0), wbar * 0.92, yerr=vals.std(0, ddof=1), color=col, label=lab,
           error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5, ecolor=INK2))
ax.set_xticks(xs); ax.set_xticklabels(['WBC', 'RBC', 'Platelet', 'Mean'])
for t, c in zip(ax.get_xticklabels(), [COL['WBC'], COL['RBC'], COL['Platelets'], INK]): t.set_color(c)
ax.set_ylim(75, 101); ax.set_ylabel('AP50–95 (%)'); tidy(ax)
ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 0.99), ncol=3, fontsize=5.6, handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
panel_label(ax, 'c', x=-0.17)
# d: example crops with the annotation and the detector box
G = load_gt('test'); P = load_pred('yolo26m_s0', 'test')
EX = [('45f8adfa34315e869de0f752de0f2cda', 2, 0.678), ('1299047edeb15e2581d9680aaa667d5d', 2, 0.825), ('846ec3b76cbc5f2d9572e454cfbc589b', 2, 0.854),
      ('4413965d02685f829ddd56f71d0c3290', 1, 0.533), ('940b907e4df95624a2ab11407a289ebb', 1, 0.542), ('9bd635d2299d5dd582cd72cfac519168', 1, 0.626)]
def pick(stem, c, target):
    p = P[stem]; gb, gc = G[stem]; m = p['conf'] >= 0.5
    b, cl = p['box'][m], p['cls'][m]
    iou = box_iou(b, gb) * (cl[:, None] == gc[None, :]); j = iou.argmax(1); best = iou.max(1)
    cand = np.where((cl == c) & (best >= 0.5))[0]
    i = cand[np.argmin(np.abs(best[cand] - target))]
    return b[i], gb[j[i]], best[i]
x0 = 0.06; wcrop = 0.135; gap = 0.023
fig.text(0.02, 0.352, 'd', fontsize=8.5, fontweight='bold', color=INK, va='bottom')
fig.text(0.06, 0.352, 'Low-IoU detections of YOLO26-m on the test set: annotation (colored box) and detector box (dashed)',
         fontsize=6.5, color=INK, va='bottom')
for n, (stem, c, tgt) in enumerate(EX):
    b, g, iou = pick(stem, c, tgt)
    u = np.concatenate([np.minimum(b[:2], g[:2]), np.maximum(b[2:], g[2:])])
    half = max(u[2] - u[0], u[3] - u[1]) * 0.75; cx, cy = (u[0] + u[2]) / 2, (u[1] + u[3]) / 2
    Wi, Hi = IMG.loc[stem + '.png', ['W', 'H']]
    cx = min(max(cx, half), Wi - half); cy = min(max(cy, half), Hi - half)  # keep the window inside the image
    reg = np.array([cx - half, cy - half, cx + half, cy + half])
    im = Image.open(f'TXL-PBC_Dataset-master/TXL-PBC/images/test/{stem}.png').convert('RGB')
    cr = im.transform((400, 400), Image.EXTENT, tuple(reg), Image.BICUBIC, fillcolor=(255, 255, 255))
    ax = fig.add_axes([x0 + n * (wcrop + gap), 0.07, wcrop, wcrop * 174 / 104]); ax.imshow(np.asarray(cr), extent=(0, 1, 1, 0))
    k = 1 / (reg[2] - reg[0])
    for bb, col, ls, lw in ((g, COL[CLS[c]], '-', 1.3), (b, INK, (0, (2.2, 1.4)), 0.9)):
        ax.add_patch(Rectangle(((bb[0] - reg[0]) * k, (bb[1] - reg[1]) * k), (bb[2] - bb[0]) * k, (bb[3] - bb[1]) * k, fc='none', ec=col, lw=lw, ls=ls))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(HAIR); sp.set_linewidth(0.4)
    src = IMG.loc[stem + '.png', 'src']
    ax.set_title(f'{CLSNAME[CLS[c]]}, {src}', fontsize=5.8, color=COL[CLS[c]], pad=2)
    ax.text(0.5, -0.06, f'IoU {iou:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=6, color=INK)
save(fig, 'Fig7_localization_analysis')

# ===================================================================== Fig. 8: CellRefine results
fig = plt.figure(figsize=(174 * MM, 62 * MM))
CFG = [('main', 'YOLO26-m + CellRefine', INK, 'o'), ('light', 'YOLO12-s + CellRefine-L', GREY2, 's')]
# a: paired seeds
ax = fig.add_axes([0.07, 0.15, 0.27, 0.75])
for k, (key, lab, col, mk) in enumerate(CFG):
    b = N[key]['test.mAP50-95']['base']; r = N[key]['test.mAP50-95']['ref']
    for s in range(3):
        x = k * 4 + s
        ax.plot([x, x], [b[s], r[s]], color=col, lw=0.8, zorder=2)
        ax.scatter(x, b[s], s=16, marker=mk, facecolor='white', edgecolor=BASE, lw=0.8, zorder=3)
        ax.scatter(x, r[s], s=16, marker=mk, color=col, zorder=4, lw=0)
    ax.text(k * 4 + 1, 86.95, lab.replace(' + ', ' +\n'), ha='center', va='top', fontsize=5.8, color=col)
ax.axhline(S.loc['yolo26m', 'mAP50-95_mean'] * 100, color=INK, lw=0.4, ls=(0, (2, 2)), zorder=1)
ax.text(6.85, S.loc['yolo26m', 'mAP50-95_mean'] * 100, 'best\ndetector', fontsize=5.2, color=INK2, ha='left', va='center', clip_on=False)
ax.set_xticks([0, 1, 2, 4, 5, 6]); ax.set_xticklabels(['0', '1', '2', '0', '1', '2']); ax.set_xlabel('Seed', labelpad=1)
ax.set_xlim(-0.7, 6.7); ax.set_ylim(86.6, 89.0); ax.set_ylabel('mAP50–95 on the test set (%)'); tidy(ax)
ax.scatter([], [], s=16, facecolor='white', edgecolor=BASE, lw=0.8, label='Detector'); ax.scatter([], [], s=16, color=INK, label='With CellRefine')
ax.legend(loc='upper left', ncol=2, fontsize=5.6, handletextpad=0.2, columnspacing=0.8)
panel_label(ax, 'a', x=-0.16)
# b: gain per metric
ax = fig.add_axes([0.42, 0.15, 0.25, 0.75])
mets = [('test.mAP50', 'mAP50'), ('test.mAP75', 'mAP75'), ('test.mAP50-95', 'mAP50–95')]
for k, (key, lab, col, mk) in enumerate(CFG):
    for i, (m, ml) in enumerate(mets):
        d = np.array(N[key][m]['d']); x = i + (k - 0.5) * 0.3
        ax.bar(x, d.mean(), 0.27, color=col, alpha=0.9 if k == 0 else 0.75, zorder=2)
        ax.errorbar(x, d.mean(), yerr=d.std(ddof=1), color=INK2, lw=0.5, capsize=1.2, capthick=0.5, zorder=3)
        ax.scatter(np.full(3, x) + np.array([-0.06, 0, 0.06]), d, s=5, color='white', edgecolor=col, lw=0.5, zorder=4)
ax.axhline(0, color=INK2, lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in mets])
ax.set_ylabel('Gain on the test set (points)'); ax.set_ylim(-0.25, 1.0); tidy(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for _, l, c, _ in CFG], loc='upper left', fontsize=5.6, handlelength=1.0)
panel_label(ax, 'b', x=-0.16)
# c: gain per class
ax = fig.add_axes([0.745, 0.15, 0.24, 0.75])
for i, c in enumerate(CLS):
    for k, (key, lab, col, mk) in enumerate(CFG):
        d = np.array(N[key][f'test.AP50-95_{c}']['d']); x = i + (k - 0.5) * 0.3
        ax.bar(x, d.mean(), 0.27, color=COL[c], alpha=1.0 if k == 0 else 0.45, zorder=2, edgecolor=COL[c], lw=0.5)
        ax.errorbar(x, d.mean(), yerr=d.std(ddof=1), color=INK2, lw=0.5, capsize=1.2, capthick=0.5, zorder=3)
        ax.scatter(np.full(3, x) + np.array([-0.06, 0, 0.06]), d, s=5, color='white', edgecolor=INK2, lw=0.5, zorder=4)
ax.axhline(0, color=INK2, lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(['WBC', 'RBC', 'Platelet'])
for t, c in zip(ax.get_xticklabels(), CLS): t.set_color(COL[c])
ax.set_ylabel('AP50–95 gain (points)'); ax.set_ylim(-0.4, 1.6); tidy(ax)
ax.text(0.98, 0.97, 'dark: YOLO26-m + CellRefine\nlight: YOLO12-s + CellRefine-L', transform=ax.transAxes, ha='right', va='top', fontsize=5.4, color=INK2)
panel_label(ax, 'c', x=-0.17)
save(fig, 'Fig8_cellrefine_results')

# ===================================================================== Fig. 9: training curves and design analysis (validation set)
fig = plt.figure(figsize=(174 * MM, 72 * MM))
ax = fig.add_axes([0.07, 0.17, 0.36, 0.73])
SEEDC = [INK, GREY2, '#b0afa9']
for s in range(3):
    h = N['hist'][f'r18_s{s}']
    ax.plot(h['ep'], h['ref'], color=SEEDC[s], lw=1.0, marker='o', ms=2.0, label=f'Refined boxes, seed {s}')
    ax.plot(h['ep'], h['resc'], color=SEEDC[s], lw=0.8, ls=(0, (3, 2)))
    ax.axhline(h['base'], color=SEEDC[s], lw=0.5, ls=(0, (1, 1.5)))
ax.plot([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label='IoU rescoring only (β = 0.5)')
ax.plot([], [], color=INK2, lw=0.5, ls=(0, (1, 1.5)), label='Detector without refinement')
ax.set_xlabel('Refiner training epoch'); ax.set_ylabel('mAP50–95 on the validation set (%)'); ax.set_xlim(2, 42); ax.set_ylim(84.9, 87.5)
ax.set_xticks(range(4, 41, 4)); tidy(ax, 'both'); ax.legend(loc='lower right', fontsize=5.5, ncol=1)
panel_label(ax, 'a', x=-0.11)
# b: design and inference variants (YOLO26-m seed 0 detector, validation set)
ax = fig.add_axes([0.62, 0.17, 0.36, 0.73])
H = N['hist']; G0 = N['grid_r18_s0']; GT = N['grid_tta8']; VV = N['voting']; VR = N['variants']
rows = [  # label, value, colour (stage colour of the changed component), group
 ('Crop context 1.3 ×', H['r18_c13']['best_ref'], BB), ('Crop context 1.6 × (default)', H['r18_s0']['best_ref'], BB), ('Crop context 2.0 ×', H['r18_c20']['best_ref'], BB),
 ('ResNet-18 without layer4', H['l3_s0']['best_ref'], BB), ('  + distillation from ResNet-18', H['l3kd_s0']['best_ref'], BB),
 ('100 epochs, σ max 0.15', H['r18_e100']['best_ref'], INK2), ('100 epochs, σ max 0.08', H['r18_e100_s08']['best_ref'], INK2),
 ('Two refinement passes', G0['i2_b0.0'], HD), ('IoU rescoring β = 0.25', G0['i1_b0.25'], HD), ('IoU rescoring β = 0.5', G0['i1_b0.5'], HD),
 ('Rescoring only, β = 0.5', G0['i0_b0.5'], HD), ('Flip test-time augmentation (8)', GT['i1_b0.0_l1.0'], HD),
 ('Weighted box voting (best)', max(v for k, v in VV.items() if k.startswith('val|vote')), INK2)]
y = np.arange(len(rows))[::-1]
base = H['r18_s0']['base']
for yy, (lab, v, col) in zip(y, rows):
    ax.plot([base, v], [yy, yy], color=HAIR, lw=0.6, zorder=1)
    ax.scatter(v, yy, s=14, color=col, zorder=3, lw=0)
ax.axvline(base, color=INK2, lw=0.5, ls=(0, (1, 1.5)))
ax.axvline(H['r18_s0']['best_ref'], color=INK, lw=0.4, ls=(0, (2, 2)))
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
for t, r in zip(ax.get_yticklabels(), rows):
    t.set_color(r[2] if r[2] != INK2 else INK)
ax.set_xlabel('mAP50–95 on the validation set (%)'); ax.set_xlim(86.1, 87.05); tidy(ax, 'x')
for k, (t, c) in enumerate((('crop and encoder', BB), ('training or post-processing', INK2), ('inference options', HD))):
    ax.text(86.12, len(rows) - 1 - 0.85 * k, t, color=c, fontsize=5.4, ha='left', va='center')
ax.text(base, len(rows) - 0.2, 'detector', fontsize=5.2, color=INK2, ha='center', va='bottom')
ax.text(H['r18_s0']['best_ref'], len(rows) - 0.2, 'default', fontsize=5.2, color=INK, ha='center', va='bottom')
panel_label(ax, 'b', x=-0.47)
save(fig, 'Fig9_refiner_training_design')

# ===================================================================== Fig. 10: other detectors and accuracy against model size
fig = plt.figure(figsize=(174 * MM, 78 * MM))
PL = N['plug']
order = sorted(PL, key=lambda k: PL[k]['mAP50-95']['base'])
ax = fig.add_axes([0.105, 0.14, 0.34, 0.78])
for i, k in enumerate(order):
    b, r = PL[k]['mAP50-95']['base'], PL[k]['mAP50-95']['ref']; fam = S.loc[k, 'family']
    ax.plot([b, r], [i, i], color=HAIR, lw=1.0, zorder=1)
    ax.scatter(b, i, s=14, marker=FMK[fam], facecolor='white', edgecolor=FCOL[fam], lw=0.8, zorder=3)
    ax.scatter(r, i, s=16, marker=FMK[fam], color=INK, zorder=4, lw=0)
    ax.text(88.95, i, f'+{r - b:.2f}', va='center', ha='right', fontsize=5.6, color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LABEL[k] for k in order], fontsize=5.8)
for t, k in zip(ax.get_yticklabels(), order):
    if k == 'yolo26m': t.set_fontweight('bold')
ax.set_xlim(86.3, 89.0); ax.set_xlabel('mAP50–95 on the test set (%)'); tidy(ax, 'x')
ax.scatter([], [], s=14, facecolor='white', edgecolor=MUTED, lw=0.8, label='Detector'); ax.scatter([], [], s=14, color=INK, label='With CellRefine')
ax.legend(loc='upper left', fontsize=5.6, handletextpad=0.2)
ax.set_title('Same refiner (trained on YOLO26-m) applied to 12 detectors', fontsize=6.5, loc='left', color=INK)
panel_label(ax, 'a', x=-0.2)
# b: accuracy against total parameters (means of three runs)
ax = fig.add_axes([0.565, 0.14, 0.42, 0.78])
for f in FCOL:
    kk = S.family == f
    ax.scatter(S.loc[kk, 'params_M'], S.loc[kk, 'mAP50-95_mean'] * 100, marker=FMK[f], s=11, color=FCOL[f], alpha=0.75, lw=0, label=f, zorder=2)
for key, lab, col, mk in CFG:
    r = np.array(N[key]['test.mAP50-95']['ref']); p = N[key]['det_params'] + N[key]['params']
    det = 'yolo26m' if key == 'main' else 'yolo12s'
    bx, by = S.loc[det, 'params_M'], S.loc[det, 'mAP50-95_mean'] * 100
    ax.annotate('', xy=(p, r.mean()), xytext=(bx, by), arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', color=col, lw=0.7, shrinkA=2, shrinkB=3))
    ax.errorbar(p, r.mean(), yerr=r.std(ddof=1), fmt='none', ecolor=col, elinewidth=0.6, capsize=1.3, zorder=4)
    ax.scatter(p, r.mean(), s=26, marker=mk, color=col, zorder=5, lw=0)
    ax.annotate(lab, (p, r.mean()), textcoords='offset points', xytext=(6, -1) if key == 'main' else (-6, 5), fontsize=5.6, color=col,
                ha='left' if key == 'main' else 'right', va='center' if key == 'main' else 'bottom')
ax.set_xscale('log'); ax.set_xlabel('Parameters (M), detector plus refiner'); ax.set_ylabel('mAP50–95 on the test set (%)')
ax.set_ylim(84.5, 89.0); tidy(ax, 'both')
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 0.0), ncol=5, fontsize=5.3, handletextpad=0.1, columnspacing=0.6, markerscale=1.1)
ax.set_title('Accuracy against model size (means of three runs)', fontsize=6.5, loc='left', color=INK)
panel_label(ax, 'b', x=-0.1)
save(fig, 'Fig10_other_detectors')
