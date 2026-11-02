# Fig. 2 - General structure shared by the compared YOLO detectors: input, backbone, neck, head and output.
# Colour culture: backbone violet, neck amber, head raspberry; cell classes WBC blue, RBC red, platelet green; neutral greys.
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
                     'lines.solid_capstyle': 'butt'})
CLS = {'WBC': '#0050FF', 'RBC': '#FF3300', 'Platelets': '#00D084'}
CLSNAME = {'WBC': 'White blood cell', 'RBC': 'Red blood cell', 'Platelets': 'Platelet'}
BB = dict(ink='#5B3FA0', mid='#A893D9', tint='#F3F0FA', tint2='#E4DCF4')
NK = dict(ink='#A35F00', mid='#E9B04F', tint='#FDF6E8', tint2='#F8E4BD')
HD = dict(ink='#A3245F', mid='#DE8FB4', tint='#FCF0F5', tint2='#F5D6E4')
INK, INK2, NEUT = '#1f1f1d', '#52514e', '#EEEDEA'
W, H = 174.0, 64.0
MM = 1 / 25.4
FS, FSS, FSH = 6.4, 5.8, 8.0

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

def T(x, y, s, size=FS, color=INK, ha='center', va='center', weight='normal', **k):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **k)

def rbox(x0, y0, x1, y1, fc='white', ec=INK, lw=0.5, r=0.7, ls='-', z=2):
    ax.add_patch(FancyBboxPatch((x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r, boxstyle=f'round,pad={r}',
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def arrow(p0, p1, color=INK2, lw=0.6, z=1.5):
    ax.annotate('', xy=p1, xytext=p0, zorder=z, arrowprops=dict(arrowstyle='-|>,head_length=0.32,head_width=0.16',
                color=color, lw=lw, shrinkA=0, shrinkB=0))

def line(pts, color=INK2, lw=0.6, z=1.5, ls='-'):
    pts = np.asarray(pts, float); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, ls=ls, solid_joinstyle='miter')

def header(x0, x1, y, title, color):
    T(x0, y, title, size=FSH, color=color, ha='left', weight='bold')
    ax.add_patch(Rectangle((x0, y + 1.9), x1 - x0, 0.75, fc=color, ec='none', zorder=2))

# ---------------------------------------------------------------- image
IMG = '4a58b26bf2dd5507b5c11a627772c0de.png'
im = Image.open('TXL-PBC_Dataset-master/TXL-PBC/images/test/' + IMG).convert('RGB')
iw, ih = im.size; side = min(iw, ih); im_sq = im.crop((0, 0, side, side))
bx = pd.read_csv('results/dataset/boxes.csv'); bx = bx[bx.img == IMG]
boxes = [(r.cls, (r.x - r.w / 2) * iw, (r.y - r.h / 2) * ih, r.w * iw, r.h * ih) for r in bx.itertuples()]
gray = np.asarray(im_sq.convert('L'), float) / 255.

def feat(res, st):
    g = Image.fromarray((gray * 255).astype(np.uint8)).resize((res, res), Image.BILINEAR)
    a = 1 - np.asarray(g, float) / 255.; a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    cm = LinearSegmentedColormap.from_list('r', [(0, 'white'), (0.55, st['mid']), (1, st['ink'])])
    return cm(a ** 0.9)[..., :3]

def slab(cx, cy, s, res, st, layers):
    d = 0.8
    for k in range(layers, 0, -1):
        ax.add_patch(Rectangle((cx - s / 2 + k * d, cy - s / 2 - k * d), s, s, fc=st['tint2'], ec=st['ink'], lw=0.35, zorder=2))
    ax.imshow(feat(res, st), extent=(cx - s / 2, cx + s / 2, cy + s / 2, cy - s / 2), zorder=3, interpolation='nearest')
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc='none', ec=st['ink'], lw=0.45, zorder=4))

def photo(x0, y0, s, with_boxes=False):
    ax.imshow(np.asarray(im_sq), extent=(x0, x0 + s, y0 + s, y0), zorder=3, interpolation='lanczos')
    ax.add_patch(Rectangle((x0, y0), s, s, fc='none', ec=INK2, lw=0.4, zorder=4))
    if with_boxes:
        k = s / side
        for c in ('RBC', 'WBC', 'Platelets'):
            for cc, bx0, by0, bw, bh in boxes:
                if cc != c: continue
                X0, Y0, X1, Y1 = max(bx0, 0), max(by0, 0), min(bx0 + bw, side), min(by0 + bh, side)
                if X1 <= X0 or Y1 <= Y0: continue
                ax.add_patch(Rectangle((x0 + X0 * k, y0 + Y0 * k), (X1 - X0) * k, (Y1 - Y0) * k, fc='none', ec=CLS[c],
                                       lw=0.7 if c == 'RBC' else 0.9, zorder=5, joinstyle='miter'))

YH = 3.2
R = {3: 17.0, 4: 33.0, 5: 49.0}          # rows of the three pyramid levels

# ---------------------------------------------------------------- input
header(3.0, 25.0, YH, 'Input', INK2)
photo(3.5, R[4] - 10.75, 21.5)
T(14.25, R[4] + 13.4, '640 × 640 × 3', size=FSS, color=INK2)

# ---------------------------------------------------------------- backbone
BX = (29.0, 72.5)
rbox(BX[0], 6.6, BX[1], 60.5, fc=BB['tint'], ec='none', r=1.2, z=0.5)
header(BX[0], BX[1], YH, 'Backbone', BB['ink'])
stem = [(34.0, R[3], 9.6, 160, 1, '/2'), (43.6, R[3], 8.6, 120, 2, '/4')]
pyr = [(54.0, R[3], 8.0, 80, 2, '/8'), (61.0, R[4], 6.6, 40, 3, '/16'), (67.2, R[5], 5.2, 20, 4, '/32')]
arrow((25.0, R[4]), (29.4, R[3] + 3.0), color=INK2)
for i, (cx, cy, s, res, L, lab) in enumerate(stem + pyr):
    slab(cx, cy, s, res, BB, L)
    T(cx, cy + s / 2 + 2.0, lab, size=FSS, color=BB['ink'])
seq = stem + pyr
for (a, b) in zip(seq[:-1], seq[1:]):
    if a[1] == b[1]:
        arrow((a[0] + a[2] / 2 + a[4] * 0.8 + 0.2, a[1]), (b[0] - b[2] / 2 - 0.3, b[1]), color=BB['ink'], lw=0.5)
    else:
        line([(a[0], a[1] + a[2] / 2 + 3.2), (a[0], b[1])], color=BB['ink'], lw=0.5)
        arrow((a[0], b[1]), (b[0] - b[2] / 2 - 0.3, b[1]), color=BB['ink'], lw=0.5)
for lev, (cx, cy, s, res, L, lab) in zip((3, 4, 5), pyr):
    T(cx + s / 2 + 1.6, cy - s / 2 - L * 0.8 - 1.4, f'P{lev}', size=FSS, color=BB['ink'], weight='bold', ha='left')
T(np.mean(BX), 57.6, 'Hierarchical feature extraction', size=FSS, color=BB['ink'])

# ---------------------------------------------------------------- neck
NX = (76.0, 118.0)
rbox(NX[0], 6.6, NX[1], 60.5, fc=NK['tint'], ec='none', r=1.2, z=0.5)
header(NX[0], NX[1], YH, 'Neck', NK['ink'])
TDX, BUX = 87.0, 107.0
T(TDX, 9.6, 'Top-down', size=FSS, color=NK['ink'], weight='bold'); T(BUX, 9.6, 'Bottom-up', size=FSS, color=NK['ink'], weight='bold')
def node(x, y):
    rbox(x - 3.4, y - 2.6, x + 3.4, y + 2.6, fc=NK['ink'] if False else 'white', ec=NK['ink'], lw=0.55, r=0.6, z=3)
    T(x, y, 'fuse', size=FSS, color=NK['ink'], zorder=4)
for lev in (3, 4, 5):
    node(TDX, R[lev]); node(BUX, R[lev])
# lateral inputs from the backbone
for lev, (cx, cy, s, res, L, lab) in zip((3, 4, 5), pyr):
    line([(cx + s / 2 + L * 0.8 + 0.2, cy), (TDX - 3.4 - 0.2, cy)], color=BB['ink'], lw=0.5) if lev != 3 else None
    if lev == 3:
        line([(cx + s / 2 + L * 0.8 + 0.2, cy), (TDX - 3.4 - 0.2, cy)], color=BB['ink'], lw=0.5)
    arrow((TDX - 3.6, cy), (TDX - 3.4, cy), color=BB['ink'], lw=0.5)
# top-down: P5 -> P4 -> P3 (upsampling)
arrow((TDX, R[5] - 2.6), (TDX, R[4] + 2.6), color=NK['ink'], lw=0.55); T(TDX + 1.0, (R[4] + R[5]) / 2, '×2 up', size=5.2, color=NK['ink'], ha='left')
arrow((TDX, R[4] - 2.6), (TDX, R[3] + 2.6), color=NK['ink'], lw=0.55); T(TDX + 1.0, (R[3] + R[4]) / 2, '×2 up', size=5.2, color=NK['ink'], ha='left')
# top-down -> bottom-up at each level
for lev in (3, 4, 5):
    arrow((TDX + 3.4, R[lev]), (BUX - 3.4, R[lev]), color=NK['ink'], lw=0.55)
# bottom-up: P3 -> P4 -> P5 (downsampling)
arrow((BUX, R[3] + 2.6), (BUX, R[4] - 2.6), color=NK['ink'], lw=0.55); T(BUX + 1.0, (R[3] + R[4]) / 2, '/2 down', size=5.2, color=NK['ink'], ha='left')
arrow((BUX, R[4] + 2.6), (BUX, R[5] - 2.6), color=NK['ink'], lw=0.55); T(BUX + 1.0, (R[4] + R[5]) / 2, '/2 down', size=5.2, color=NK['ink'], ha='left')
T(np.mean(NX), 57.6, 'Multi-scale feature fusion', size=FSS, color=NK['ink'])

# ---------------------------------------------------------------- head
HX = (121.5, 145.0)
rbox(HX[0], 6.6, HX[1], 60.5, fc=HD['tint'], ec='none', r=1.2, z=0.5)
header(HX[0], HX[1], YH, 'Head', HD['ink'])
hx0, hx1 = HX[0] + 1.2, HX[1] - 1.2
for lev, res in ((3, '80 × 80'), (4, '40 × 40'), (5, '20 × 20')):
    y = R[lev]
    arrow((BUX + 3.4, y), (hx0, y), color=NK['ink'], lw=0.55)
    rbox(hx0, y - 5.2, hx1, y + 5.2, fc='white', ec=HD['ink'], lw=0.55, z=3)
    T((hx0 + hx1) / 2, y - 2.8, f'P{lev} · {res}', size=FSS, color=HD['ink'], weight='bold', zorder=4)
    mw = (hx1 - hx0 - 3.0) / 2
    rbox(hx0 + 1.0, y - 0.8, hx0 + 1.0 + mw, y + 4.1, fc=HD['tint2'], ec='none', r=0.5, z=4)
    rbox(hx1 - 1.0 - mw, y - 0.8, hx1 - 1.0, y + 4.1, fc=HD['tint2'], ec='none', r=0.5, z=4)
    T(hx0 + 1.0 + mw / 2, y + 1.65, 'Box', size=FSS, zorder=5)
    T(hx1 - 1.0 - mw / 2, y + 1.65, 'Class', size=FSS, zorder=5)
T(np.mean(HX), 57.6, 'Dense predictions', size=FSS, color=HD['ink'])

# ---------------------------------------------------------------- output
OX = (148.5, 172.5)
header(OX[0], OX[1], YH, 'Output', INK2)
rbox(OX[0], 8.2, OX[1], 14.4, fc=NEUT, ec=INK2, lw=0.45, r=0.6, z=3)
T(np.mean(OX), 10.1, 'NMS or one-to-one', size=5.6, color=INK2, zorder=4); T(np.mean(OX), 12.5, 'selection', size=5.6, color=INK2, zorder=4)
for lev in (3, 4, 5):
    line([(hx1, R[lev]), (146.7, R[lev])], color=HD['ink'], lw=0.5)
line([(146.7, R[5]), (146.7, 11.3)], color=HD['ink'], lw=0.5)
arrow((146.7, 11.3), (OX[0], 11.3), color=HD['ink'], lw=0.5)
os_ = 22.5
photo(149.25, 18.8, os_, with_boxes=True)
arrow((np.mean(OX), 14.4), (np.mean(OX), 18.6), color=INK2, lw=0.5)
for i, c in enumerate(('WBC', 'RBC', 'Platelets')):
    yy = 45.4 + i * 3.3
    ax.add_patch(Rectangle((149.4, yy - 1.0), 2.0, 2.0, fc='none', ec=CLS[c], lw=0.9, zorder=4))
    T(152.6, yy, CLSNAME[c], size=FSS, ha='left')
T(np.mean(OX), 57.6, 'Boxes, classes, scores', size=FSS, color=INK2)

out = 'figures/output/Fig2_YOLO_general'
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
Image.open(out + '.png').convert('RGB').save(out + '.tiff', compression='tiff_lzw', dpi=(600, 600))
print('saved', out)
