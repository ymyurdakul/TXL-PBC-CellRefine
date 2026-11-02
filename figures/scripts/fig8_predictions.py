# Fig. 8 - Test-set predictions of YOLO26-m + CellRefine (run 0) with annotated boxes and enlarged details
# showing the detector box (dashed) and the refined box (solid) against the annotation (black).
import sys, os, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
sys.path.insert(0, 'analysis')
import evalcore as E

plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
                     'lines.solid_capstyle': 'butt'})
CLS = {0: '#0050FF', 1: '#FF3300', 2: '#00D084'}
CN = {0: 'White blood cell', 1: 'Red blood cell', 2: 'Platelet'}
INK, INK2, MUTED, HAIR, NEUT = '#1f1f1d', '#52514e', '#8a8985', '#c9c8c3', '#EEEDEA'
MM = 1 / 25.4
DASH = (0, (2.4, 1.4))

R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
PAIRS = pd.read_csv('results/analysis/pairs.csv'); PAIRS = PAIRS[PAIRS.seed == 0]
G = E.load_gt('test')
IM = pd.read_csv('results/dataset/images.csv').set_index('img')
ROWS = [('bac60b0b98055c6ca9ab5ec0b6e4ec3a', [25, 23]),     # BCCD: two platelets
        ('940b907e4df95624a2ab11407a289ebb', [10, 18]),     # BCDD: overlapping and border red cells
        ('579f9dd49c365f668b1de414493488a2', [16, 14]),     # PBC: red cell improved, platelet worsened
        ('85ed20f54a4b504fabb0fd4d67300fae', [5, 3])]       # Raabin-WBC: border red cell, interior red cell

h = 31.0; pitch = h + 6.6; TOP = 7.5
W, H = 174.0, TOP + 4 * pitch + 6.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
def T(x, y, s, size=6.2, color=INK, ha='center', va='center', weight='normal', **k):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **k)
def box(b, region, x0, y0, sx, sy, color, lw, ls='-', z=5, clip=None):
    X0, Y0 = x0 + (b[0] - region[0]) * sx, y0 + (b[1] - region[1]) * sy
    X1, Y1 = x0 + (b[2] - region[0]) * sx, y0 + (b[3] - region[1]) * sy
    if clip is not None:
        X0, Y0, X1, Y1 = max(X0, clip[0]), max(Y0, clip[1]), min(X1, clip[2]), min(Y1, clip[3])
    if X1 > X0 and Y1 > Y0:
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc='none', ec=color, lw=lw, ls=ls, zorder=z, joinstyle='miter'))

C1, C2, C3, C4 = 6.5, 56.0, 106.0, 140.5     # column starts
SLOT = 46.0; DW = 33.0
T(C1 + SLOT / 2, TOP - 3.6, 'Annotation', size=7.0, weight='bold')
T(C2 + SLOT / 2, TOP - 3.6, 'YOLO26-m + CellRefine', size=7.0, weight='bold')
T((C3 + C4 + DW) / 2, TOP - 3.6, 'Details 1 and 2', size=7.0, weight='bold')

for r, (stem, gis) in enumerate(ROWS):
    y0 = TOP + r * pitch
    src = IM.loc[stem + '.png', 'src']
    im = Image.open(f'TXL-PBC_Dataset-master/TXL-PBC/images/test/{stem}.png').convert('RGB'); iw, ih = im.size
    key = [k for k in R[(0, 'test')]['base'] if stem in k][0]
    p0, p1 = R[(0, 'test')]['base'][key], R[(0, 'test')]['ref'][key]
    keep = p0['conf'] >= 0.25
    b0, b1, cl = p0['box'][keep], p1['box'][keep], p0['cls'][keep]
    gb, gc = G[stem]
    T(2.6, y0 + h / 2, src, size=6.6, rotation=90, weight='bold', color=INK2)
    wimg = h * iw / ih
    full = (0, 0, iw, ih)
    for c0, boxes, classes in ((C1, gb, gc), (C2, b1, cl)):
        x0 = c0 + (SLOT - wimg) / 2
        ax.imshow(np.asarray(im), extent=(x0, x0 + wimg, y0 + h, y0), zorder=3, interpolation='lanczos')
        ax.add_patch(Rectangle((x0, y0), wimg, h, fc='none', ec=INK2, lw=0.4, zorder=6))
        for c in (1, 0, 2):
            for b in boxes[classes == c]:
                box(b, full, x0, y0, wimg / iw, h / ih, CLS[c], 0.5 if c == 1 else 0.75, clip=(x0, y0, x0 + wimg, y0 + h))
        if c0 == C2: xp = x0
    for j, gi in enumerate(gis):
        pr = PAIRS[(PAIRS.img == stem) & (PAIRS.gi == gi)].iloc[0]
        pi = int(pr.pi); c = int(gc[gi]); g = gb[gi]; d0, d1 = b0[pi], b1[pi]
        allb = np.vstack([g, d0, d1])
        cx, cy = (allb[:, 0].min() + allb[:, 2].max()) / 2, (allb[:, 1].min() + allb[:, 3].max()) / 2
        side = 1.45 * max(allb[:, 2].max() - allb[:, 0].min(), allb[:, 3].max() - allb[:, 1].min())
        reg = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
        # crop with neutral padding outside the image
        canvas = Image.new('RGB', (iw + 2 * int(side), ih + 2 * int(side)), (238, 237, 234)); o = int(side)
        canvas.paste(im, (o, o))
        cr = canvas.transform((600, 600), Image.EXTENT, tuple(reg + o), Image.BICUBIC)
        xd = C3 if j == 0 else C4
        ax.imshow(np.asarray(cr), extent=(xd, xd + DW, y0 + h, y0), zorder=3, interpolation='lanczos')
        ax.add_patch(Rectangle((xd, y0), DW, h, fc='none', ec=INK2, lw=0.4, zorder=7))
        s = DW / side; sy = h / side
        clip = (xd, y0, xd + DW, y0 + h)
        box(g, reg, xd, y0, s, sy, INK, 0.6, z=5, clip=clip)
        box(d0, reg, xd, y0, s, sy, CLS[c], 1.0, ls=DASH, z=6, clip=clip)
        box(d1, reg, xd, y0, s, sy, CLS[c], 1.0, z=6, clip=clip)
        T(xd + 1.6, y0 + 2.2, str(j + 1), size=6.6, weight='bold', ha='left', zorder=8,
          bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
        T(xd + DW / 2, y0 + h + 2.6, f'{CN[c]}, IoU {pr.iou0:.2f} → {pr.iou1:.2f}', size=5.8, color=INK)
        # mark the detail on the prediction panel
        sxp, syp = wimg / iw, h / ih
        m = 0.35 * side
        mb = np.array([g[0] - m * 0.25, g[1] - m * 0.25, g[2] + m * 0.25, g[3] + m * 0.25])
        X0 = max(xp + mb[0] * sxp, xp); Y0 = max(y0 + mb[1] * syp, y0)
        X1 = min(xp + mb[2] * sxp, xp + wimg); Y1 = min(y0 + mb[3] * syp, y0 + h)
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc='none', ec=INK, lw=0.6, ls=(0, (1.2, 0.8)), zorder=8))
        lx = X0 - 0.4 if X0 - xp > 3 else X1 + 0.4
        T(lx, Y0 + 1.2, str(j + 1), size=6.0, weight='bold', ha='right' if X0 - xp > 3 else 'left', zorder=9,
          bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

# legend
yl = TOP + 4 * pitch + 1.2
hs = [Line2D([0], [0], color=INK, lw=0.8, label='Annotation'),
      Line2D([0], [0], color=INK2, lw=1.0, ls=(0, (2.4, 1.4)), label='YOLO26-m'),
      Line2D([0], [0], color=INK2, lw=1.0, label='YOLO26-m + CellRefine')] + \
     [Line2D([0], [0], marker='s', ls='none', markerfacecolor='none', markeredgecolor=CLS[c], markeredgewidth=1.0, markersize=5, label=CN[c]) for c in (0, 1, 2)]
fig.legend(handles=hs, loc='center', bbox_to_anchor=(0.5, 1 - (yl + 1.6) / H), ncol=6, frameon=False, fontsize=6.2,
           handlelength=2.2, columnspacing=1.4, handletextpad=0.5)
out = 'figures/output/Fig8_predictions'
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
Image.open(out + '.png').convert('RGB').save(out + '.tiff', compression='tiff_lzw', dpi=(600, 600))
print('H', H)
