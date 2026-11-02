# Manuscript figures 3-7: benchmark results on TXL-PBC and the analysis of the best model (YOLO26-m, seed 0).
import sys, json, glob, os, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
from PIL import Image
sys.path.insert(0, 'figures/scripts')
from style import COL, INK, INK2, MUTED, GRID, HAIR, MM  # paper palette and rcParams
plt.rcParams.update({'font.size': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6})

OUT = 'figures/output'; RES = 'results/benchmark'; DATA = 'TXL-PBC_Dataset-master/TXL-PBC'
CLSNAME = {'WBC': 'White blood cell', 'RBC': 'Red blood cell', 'Platelets': 'Platelet'}
CLS = ['WBC', 'RBC', 'Platelets']
df = pd.read_csv(f'{RES}/runs_long.csv')
S = pd.read_csv(f'{RES}/summary_test.csv', index_col=0).sort_values('rank')
FAMS = ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLO11', 'YOLO12', 'YOLO26', 'RT-DETR']
FCOL = {'YOLOv3': '#9a8c98', 'YOLOv5': '#6d8fa6', 'YOLOv6': '#b08968', 'YOLOv8': '#7a9e7e', 'YOLOv9': '#8d7bb0',
        'YOLOv10': '#c38d9e', 'YOLO11': '#5f7f9a', 'YOLO12': '#a67c52', 'YOLO26': '#3a3a3a', 'RT-DETR': '#b5651d'}
FMK = {'YOLOv3': 'v', 'YOLOv5': 's', 'YOLOv6': 'P', 'YOLOv8': 'D', 'YOLOv9': '^', 'YOLOv10': 'X', 'YOLO11': 'o',
       'YOLO12': 'h', 'YOLO26': '*', 'RT-DETR': 'd'}
BEST = 'yolo26m'
LABEL = {m: (f if v == 'base' else f'{f}-{v}') for m, f, v in zip(S.index, S.family, S.variant)}

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=600); fig.savefig(f'{OUT}/{name}.pdf')
    Image.open(f'{OUT}/{name}.png').convert('RGB').save(f'{OUT}/{name}.tiff', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig); print('saved', name)

def panel_label(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', ha='right', va='bottom', color=INK)

def fam_sep(ax, horizontal, y):
    fams = list(S.family); prev = fams[0]
    for i, f in enumerate(fams):
        if f != prev:
            (ax.axhline(y[i] + 0.5, color=HAIR, lw=0.4, zorder=1) if horizontal else ax.axvline(i - 0.5, color=HAIR, lw=0.4, zorder=1)); prev = f

# ================================================================ Fig. 3: benchmark overview
fig = plt.figure(figsize=(174 * MM, 150 * MM))
ax = fig.add_axes([0.115, 0.075, 0.39, 0.905])
y = np.arange(len(S))[::-1]
for i, (m, r) in enumerate(S.iterrows()):
    c = INK if m == BEST else FCOL[r.family]
    ax.errorbar(r['mAP50-95_mean'] * 100, y[i], xerr=r['mAP50-95_sd'] * 100, fmt='none', ecolor=c, elinewidth=0.7, capsize=1.3, capthick=0.5, zorder=2)
    ax.scatter(r['mAP50-95_mean'] * 100, y[i], marker=FMK[r.family], s=22 if m == BEST else 11, color=c, zorder=3, linewidths=0)
    for v in df[(df.model == m) & (df.split == 'test')]['mAP50-95']:
        ax.scatter(v * 100, y[i], marker='|', s=9, color=c, alpha=0.6, zorder=2, linewidths=0.5)
ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in S.index], fontsize=5.6)
for lab, m in zip(ax.get_yticklabels(), S.index):
    if m == BEST: lab.set_fontweight('bold')
ax.set_ylim(-0.8, len(S) - 0.2); ax.set_xlim(84.5, 88.6)
ax.grid(axis='x', color=GRID, lw=0.4, zorder=0); ax.set_axisbelow(True); ax.set_xlabel('mAP50–95 (%)')
fam_sep(ax, True, y)
ax.axvline(S.loc[BEST, 'mAP50-95_mean'] * 100, color=INK, lw=0.4, ls=(0, (2, 2)), zorder=1)
panel_label(ax, 'a', x=-0.27, y=1.0)
# right column
def pareto(x, yv):
    idx = np.argsort(x); best = -1; keep = []
    for i in idx:
        if yv[i] > best: keep.append(i); best = yv[i]
    return keep
for k, (xcol, xl, pos) in enumerate((('GFLOPs', 'GFLOPs', [0.60, 0.565, 0.385, 0.405]), ('FPS_mean', 'Throughput (images/s)', [0.60, 0.075, 0.385, 0.405]))):
    ax = fig.add_axes(pos)
    x = S[xcol].values.astype(float); yv = S['mAP50-95_mean'].values * 100; e = S['mAP50-95_sd'].values * 100
    for f in FAMS:
        kk = (S.family == f).values
        ax.errorbar(x[kk], yv[kk], yerr=e[kk], fmt='none', ecolor=FCOL[f], elinewidth=0.45, alpha=0.6, zorder=2)
        ax.scatter(x[kk], yv[kk], marker=FMK[f], s=13 if f != 'YOLO26' else 20, color=FCOL[f], label=f, zorder=3, linewidths=0)
    ib = list(S.index).index(BEST)
    ax.scatter(x[ib], yv[ib], marker='*', s=40, color=INK, zorder=4, linewidths=0)
    keep = sorted(pareto(x if xcol == 'GFLOPs' else -x, yv), key=lambda i: x[i])
    ax.plot(x[keep], yv[keep], color=INK, lw=0.5, ls=(0, (2, 2)), zorder=1)
    for i in keep:
        right = xcol != 'GFLOPs'
        ax.annotate(LABEL[S.index[i]], (x[i], yv[i]), textcoords='offset points', xytext=((-3, 3) if right else (3, 3)),
                    ha='right' if right else 'left', fontsize=5, color=INK)
    ax.set_xscale('log'); ax.set_xlabel(xl); ax.set_ylabel('mAP50–95 (%)'); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.set_ylim(84.6, 88.6)
    panel_label(ax, 'bc'[k], x=-0.13, y=1.0)
    if k == 1:
        ax.legend(ncol=2, loc='lower left', fontsize=5.2, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2, markerscale=0.9)
save(fig, 'Fig3_benchmark_overview')

# ================================================================ Fig. 4: class-wise AP
fig, axs = plt.subplots(2, 1, figsize=(174 * MM, 100 * MM), sharex=True)
x = np.arange(len(S))
for ax, col, lab in ((axs[0], 'AP50-95', 'a'), (axs[1], 'AP50', 'b')):
    for c in CLS:
        m = S[f'{col}_{c}_mean'].values * 100; sd = S[f'{col}_{c}_sd'].values * 100
        ax.fill_between(x, m - sd, m + sd, color=COL[c], alpha=0.15, lw=0)
        ax.plot(x, m, color=COL[c], lw=0.9, marker='o', ms=2.0, label=CLSNAME[c])
    ax.set_ylabel(f'{col.replace("-", "–")} (%)'); ax.grid(axis='y', color=GRID, lw=0.4); ax.set_axisbelow(True)
    fam_sep(ax, False, None); panel_label(ax, lab, x=-0.06, y=1.0)
    ib = list(S.index).index(BEST); ax.axvline(ib, color=INK, lw=0.4, ls=(0, (2, 2)), zorder=1)
axs[0].legend(ncol=3, loc='lower right', fontsize=5.8)
axs[1].set_xticks(x); axs[1].set_xticklabels([LABEL[m] for m in S.index], rotation=90, fontsize=5.2)
for lab, m in zip(axs[1].get_xticklabels(), S.index):
    if m == BEST: lab.set_fontweight('bold')
axs[0].set_ylim(76, 93); axs[1].set_ylim(94, 100)
fig.subplots_adjust(left=0.075, right=0.995, top=0.96, bottom=0.21, hspace=0.22)
save(fig, 'Fig4_classwise_ap')

# ================================================================ Fig. 5: best model - confusion matrices and curves
cm = np.load(f'{RES}/out/best_cm25.npy'); cur = np.load(f'{RES}/out/best_curves.npz')
xq, pc, rc, f1c, pv, ap, nt, conf_best = cur['x'], cur['pc'], cur['rc'], cur['f1c'], cur['prec_values'], cur['ap'], cur['nt'], float(cur['conf_best'])
fig, axs = plt.subplots(2, 3, figsize=(174 * MM, 112 * MM))
NAMES4 = ['White blood cell', 'Red blood cell', 'Platelet', 'Background']
CC4 = [COL['WBC'], COL['RBC'], COL['Platelets'], INK2]
def draw_cm(ax, M, normalised):
    n = 4
    colsum = M.sum(0).astype(float)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            frac = (v / colsum[j]) if (colsum[j] > 0 and j < 3) else (v / M[:, 3].sum() if M[:, 3].sum() > 0 else 0)
            if i == j and i < 3:
                base = np.array(to_rgb(CC4[i])); fc = 1 - (1 - base) * (0.15 + 0.85 * frac)
            else:
                fc = np.array([1, 1, 1]) - (1 - np.array(to_rgb('#52514e'))) * (0.9 * frac) if v > 0 else np.array([1, 1, 1])
            ax.add_patch(Rectangle((j, i), 1, 1, fc=fc, ec='white', lw=1.2))
            if (v > 0 or (i == j)) and not (i == 3 and j == 3):
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                txt = f'{frac * 100:.1f}' if normalised else f'{v}'
                if normalised and j == 3: txt = f'{v}'  # background column has no column total: show counts
                ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=6.6 if not normalised else 6.4,
                        color='white' if lum < 0.55 else INK, fontweight='bold' if (i == j and i < 3) else 'normal')
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect('equal')
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(NAMES4, rotation=35, ha='right', fontsize=5.6); ax.set_yticklabels(NAMES4, fontsize=5.6)
    for t, c in zip(ax.get_xticklabels(), CC4): t.set_color(c)
    for t, c in zip(ax.get_yticklabels(), CC4): t.set_color(c)
    ax.tick_params(length=0); ax.set_xlabel('True class'); ax.set_ylabel('Predicted class')
    for sp in ax.spines.values(): sp.set_visible(False)
draw_cm(axs[0, 0], cm, False); panel_label(axs[0, 0], 'a', x=-0.42, y=1.0)
axs[0, 0].set_title('Counts, confidence ≥ 0.25', fontsize=6.5, loc='left', pad=3)
draw_cm(axs[0, 1], cm, True); panel_label(axs[0, 1], 'b', x=-0.42, y=1.0)
axs[0, 1].set_title('Percentage of each true class', fontsize=6.5, loc='left', pad=3)
# PR curves
ax = axs[0, 2]
for ci, c in enumerate(CLS):
    ax.plot(xq, pv[ci], color=COL[c], lw=0.9, label=f'{CLSNAME[c]} ({ap[ci, 0] * 100:.1f})')
ax.plot(xq, pv.mean(0), color=INK, lw=1.1, label=f'All classes ({ap[:, 0].mean() * 100:.1f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True); ax.legend(loc='lower left', fontsize=5.4, title='AP50 (%)', title_fontsize=5.6)
ax.set_title('Precision–recall at IoU 0.5', fontsize=6.5, loc='left', pad=3); panel_label(ax, 'c', x=-0.2, y=1.0)
# P, R, F1 vs confidence
for k, (arr, yl, lab) in enumerate(((pc, 'Precision', 'd'), (rc, 'Recall', 'e'), (f1c, 'F1-score', 'f'))):
    ax = axs[1, k]
    for ci, c in enumerate(CLS):
        ax.plot(xq, arr[ci], color=COL[c], lw=0.8, label=CLSNAME[c])
    ax.plot(xq, arr.mean(0), color=INK, lw=1.1, label='All classes')
    ax.axvline(conf_best, color=INK, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel('Confidence threshold'); ax.set_ylabel(yl); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True); panel_label(ax, lab, x=-0.2, y=1.0)
    if k == 2:
        ax.annotate(f'max mean F1 = {f1c.mean(0)[int(cur["ib"])]:.3f}\nat confidence {conf_best:.2f}', (conf_best, 0.35), textcoords='offset points',
                    xytext=(-6, 0), ha='right', fontsize=5.4, color=INK)
        ax.legend(loc='lower left', fontsize=5.4)
fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.1, wspace=0.5, hspace=0.62)
save(fig, 'Fig5_best_model_evaluation')

# ================================================================ Fig. 6: training curves of the best model (3 seeds)
U = 'TXLPBC_runs'
def run_dir(model, seed):
    if seed == 0: return f'{U}/train/{model}'
    if model == 'yolo26m': return f'{U}/train_cell/base_yolo26m_s{seed}'
    return f'{U}/seeds/train/{model}_s{seed}'
SEEDCOL = ['#1f1f1d', '#7a7976', '#b8b7b2']
fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 55 * MM))
for s in (0, 1, 2):
    d = pd.read_csv(run_dir(BEST, s) + '/results.csv'); d.columns = d.columns.str.strip(); ep = d['epoch']
    axs[0].plot(ep, d['train/box_loss'], color=SEEDCOL[s], lw=0.8, label='Training' if s == 0 else None)
    axs[0].plot(ep, d['val/box_loss'], color=SEEDCOL[s], lw=0.8, ls=(0, (3, 2)), label='Validation' if s == 0 else None)
    axs[1].plot(ep, d['train/cls_loss'], color=SEEDCOL[s], lw=0.8, label=f'Seed {s}')
    axs[1].plot(ep, d['val/cls_loss'], color=SEEDCOL[s], lw=0.8, ls=(0, (3, 2)))
    axs[2].plot(ep, d['metrics/mAP50(B)'] * 100, color=SEEDCOL[s], lw=0.8, ls=(0, (3, 2)))
    axs[2].plot(ep, d['metrics/mAP50-95(B)'] * 100, color=SEEDCOL[s], lw=0.9, label=f'Seed {s}')
axs[0].set_ylabel('Box loss'); axs[1].set_ylabel('Classification loss'); axs[2].set_ylabel('Validation mAP (%)')
axs[2].set_ylim(60, 100)
axs[0].legend(fontsize=5.4, loc='upper right'); axs[1].legend(fontsize=5.4, loc='upper right'); axs[2].legend(fontsize=5.4, loc='lower right')
axs[2].text(0.97, 0.72, 'dashed: mAP50\nsolid: mAP50–95', transform=axs[2].transAxes, ha='right', fontsize=5.4, color=INK2)
for ax, lab in zip(axs, 'abc'):
    ax.set_xlabel('Epoch'); ax.set_xlim(0, 100); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True); panel_label(ax, lab, x=-0.2, y=1.0)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.18, wspace=0.38)
save(fig, 'Fig6_best_training_curves')

# ================================================================ Fig. 7: qualitative predictions (GT vs YOLO26-m)
B = pd.read_csv('results/dataset/boxes.csv'); IMG = pd.read_csv('results/dataset/images.csv').set_index('img')
preds = json.load(open(f'{U}/eval_test/yolo26m/predictions.json'))
PB = {}
for d in preds:
    x0, y0, w, h = d['bbox']; PB.setdefault(d['image_id'], []).append((CLS[d['category_id'] - 1], x0, y0, x0 + w, y0 + h, d['score']))
PICK = [('BCCD', 'bac60b0b98055c6ca9ab5ec0b6e4ec3a.png'), ('BCDD', '40b9e4b0f2c652b281395e5c40f89a3d.png'),
        ('PBC', '4a58b26bf2dd5507b5c11a627772c0de.png'), ('Raabin-WBC', 'a581fbea576e5e1196baaa0b37aa7406.png')]
def load_square(img):
    im = cv2.cvtColor(cv2.imread(f'{DATA}/images/test/{img}'), cv2.COLOR_BGR2RGB); H, W = im.shape[:2]
    g = B[B.img == img].copy()
    g['x1'] = (g.x - g.w / 2) * W; g['x2'] = (g.x + g.w / 2) * W; g['y1'] = (g.y - g.h / 2) * H; g['y2'] = (g.y + g.h / 2) * H
    Ssz = min(W, H)
    if W == H: ox = oy = 0
    elif W > H:
        oy = 0; best = None
        for ox in range(0, W - Ssz + 1, 4):
            inside = ((g.x1 >= ox) & (g.x2 <= ox + Ssz)); sc = (inside * (g.cls.map({'WBC': 20, 'Platelets': 4, 'RBC': 0.2}))).sum()
            if best is None or sc > best[0] + 1e-9: best = (sc, ox)
        ox = best[1]
    else:
        ox = 0; oy = (H - Ssz) // 2
    return im[oy:oy + Ssz, ox:ox + Ssz], g, Ssz, ox, oy
LW = {'WBC': 1.3, 'RBC': 0.9, 'Platelets': 1.3}
def draw_boxes(ax, boxes, Ssz, ox, oy, ls='-'):
    for c, x1, y1, x2, y2 in sorted(boxes, key=lambda b: {'RBC': 0, 'WBC': 1, 'Platelets': 2}[b[0]]):
        x1, x2 = np.clip([x1 - ox, x2 - ox], 0, Ssz); y1, y2 = np.clip([y1 - oy, y2 - oy], 0, Ssz)
        if (x2 - x1) * (y2 - y1) <= 0: continue
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COL[c], lw=LW[c], ls=ls, joinstyle='miter', zorder=3 if c != 'RBC' else 2))
fig, axs = plt.subplots(2, 4, figsize=(174 * MM, 92 * MM))
for k, (src, img) in enumerate(PICK):
    crop, g, Ssz, ox, oy = load_square(img)
    gt = [(r.cls, r.x1, r.y1, r.x2, r.y2) for r in g.itertuples()]
    pr = [(c, x1, y1, x2, y2) for c, x1, y1, x2, y2, sc in PB.get(img[:-4], []) if sc >= 0.25]
    for row, boxes in ((0, gt), (1, pr)):
        ax = axs[row, k]; ax.imshow(crop, extent=[0, Ssz, Ssz, 0], interpolation='lanczos')
        draw_boxes(ax, boxes, Ssz, ox, oy)
        ax.set_xlim(0, Ssz); ax.set_ylim(Ssz, 0); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    axs[0, k].set_title(src, fontsize=7, loc='left', pad=3, fontweight='bold')
    n_gt = len(gt); n_pr = len(pr)
    axs[1, k].text(0.0, -0.035, f'{n_pr} predicted, {n_gt} annotated', transform=axs[1, k].transAxes, fontsize=5.8, color=INK2, va='top', ha='left')
axs[0, 0].set_ylabel('Ground truth', fontsize=7); axs[1, 0].set_ylabel('YOLO26-m', fontsize=7)
# legend
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=COL[c], lw=1.4, label=CLSNAME[c]) for c in CLS]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6)
fig.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.115, wspace=0.04, hspace=0.1)
save(fig, 'Fig7_qualitative')
