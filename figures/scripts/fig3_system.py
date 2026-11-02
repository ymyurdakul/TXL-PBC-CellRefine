# Fig. 3 - YOLO26-m with the CellRefine second stage: (a) inference pipeline, (b) CellRefine network, (c) training.
# Colour culture: classes WBC #0050FF, RBC #FF3300, platelet #00D084; stages backbone violet, neck amber, head raspberry;
# existing (frozen) modules white with a stage-coloured frame, proposed modules filled with the dark stage colour.
import sys, os, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from PIL import Image
sys.path.insert(0, 'analysis')
import evalcore as E

plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
                     'lines.solid_capstyle': 'butt', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})
CLS = {0: '#0050FF', 1: '#FF3300', 2: '#00D084'}
BB = dict(ink='#5B3FA0', mid='#A893D9', tint='#F3F0FA', tint2='#E4DCF4')
NK = dict(ink='#A35F00', mid='#E9B04F', tint='#FDF6E8', tint2='#F8E4BD')
HD = dict(ink='#A3245F', mid='#DE8FB4', tint='#FCF0F5', tint2='#F5D6E4')
INK, INK2, MUTED, HAIR, NEUT = '#1f1f1d', '#52514e', '#8a8985', '#c9c8c3', '#EEEDEA'
W, H = 174.0, 104.0
MM = 1 / 25.4
FS, FSS, FSH = 6.4, 5.8, 7.6
DASH = (0, (2.2, 1.3))

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

def T(x, y, s, size=FS, color=INK, ha='center', va='center', weight='normal', **k):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **k)

def rbox(x0, y0, x1, y1, fc='white', ec=INK, lw=0.5, r=0.7, ls='-', z=2):
    ax.add_patch(FancyBboxPatch((x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r, boxstyle=f'round,pad={r}',
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def filled(x0, y0, x1, y1, lines, st, size=FS, lh=2.6):
    rbox(x0, y0, x1, y1, fc=st['ink'], ec=st['ink'], lw=0.55, z=3)
    cy = (y0 + y1) / 2 - (len(lines) - 1) * lh / 2
    for i, s in enumerate(lines):
        T((x0 + x1) / 2, cy + i * lh, s, size=size, color='white', zorder=4, weight='bold' if i == 0 else 'normal')

def framed(x0, y0, x1, y1, lines, ec=INK2, fc='white', ls='-', size=FS, color=INK, lh=2.6, bold_first=False):
    rbox(x0, y0, x1, y1, fc=fc, ec=ec, lw=0.55, ls=ls, z=3)
    cy = (y0 + y1) / 2 - (len(lines) - 1) * lh / 2
    for i, s in enumerate(lines):
        T((x0 + x1) / 2, cy + i * lh, s, size=size, color=color, zorder=4, weight='bold' if (bold_first and i == 0) else 'normal')

def arrow(p0, p1, color=INK2, lw=0.6, z=1.5):
    ax.annotate('', xy=p1, xytext=p0, zorder=z, arrowprops=dict(arrowstyle='-|>,head_length=0.32,head_width=0.16',
                color=color, lw=lw, shrinkA=0, shrinkB=0))

def line(pts, color=INK2, lw=0.6, ls='-', z=1.5):
    pts = np.asarray(pts, float); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, zorder=z, solid_joinstyle='miter')

def header(x0, x1, y, title, color):
    T(x0, y, title, size=FSH, color=color, ha='left', weight='bold')
    ax.add_patch(Rectangle((x0, y + 1.8), x1 - x0, 0.7, fc=color, ec='none', zorder=2))

def panel(x, y, s):
    T(x, y, s, size=9, weight='bold', ha='left')

# ---------------------------------------------------------------- data: PBC test image, YOLO26-m run 0, before/after CellRefine
STEM = '45bc2bb9c9bd57688c652db42de77007'
im = Image.open(f'TXL-PBC_Dataset-master/TXL-PBC/images/test/{STEM}.png').convert('RGB')
iw, ih = im.size
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
key = [k for k in R[(0, 'test')]['base'] if STEM in k][0]
P0, P1 = R[(0, 'test')]['base'][key], R[(0, 'test')]['ref'][key]
keep = P0['conf'] >= 0.25
b0, b1, cl = P0['box'][keep], P1['box'][keep], P0['cls'][keep]
G = E.load_gt('test')[STEM]
def best_gt(b, c):
    m = E.box_iou(G[0], b[None])[:, 0] * (G[1] == c); j = m.argmax(); return G[0][j], m[j]
# example crops: one WBC, one RBC, the platelet with the largest improvement
ex = {}
for c in (0, 1, 2):
    idx = np.where(cl == c)[0]
    gains = [E.box_iou(best_gt(b0[i], c)[0][None], b1[i][None])[0, 0] - best_gt(b0[i], c)[1] for i in idx]
    if c in (0, 1):  # a fully visible cell near the centre
        cen = np.array([np.hypot((b0[i, 0] + b0[i, 2]) / 2 - iw / 2, (b0[i, 1] + b0[i, 3]) / 2 - ih / 2) for i in idx])
        inside = np.array([(b0[i, 0] > 5) and (b0[i, 1] > 5) and (b0[i, 2] < iw - 5) and (b0[i, 3] < ih - 5) for i in idx])
        cen[~inside] = 1e9; ex[c] = idx[cen.argmin()]
    else:
        ex[c] = idx[int(np.argmax(gains))]
ipl = ex[2]; gpl, iou0 = best_gt(b0[ipl], 2); iou1 = E.box_iou(gpl[None], b1[ipl][None])[0, 0]
print('platelet IoU', round(iou0, 3), '->', round(iou1, 3), 'n boxes', len(b0))
CTX = 1.6
def ctx_box(b, c=CTX):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; w, h = (b[2] - b[0]) * c / 2, (b[3] - b[1]) * c / 2
    return np.array([cx - w, cy - h, cx + w, cy + h])
def crop(b, size=512):
    return im.transform((size, size), Image.EXTENT, tuple(b), Image.BICUBIC)
def draw_img(x0, y0, s, img):
    ax.imshow(np.asarray(img), extent=(x0, x0 + s, y0 + s, y0), zorder=3, interpolation='lanczos')
    ax.add_patch(Rectangle((x0, y0), s, s, fc='none', ec=INK2, lw=0.4, zorder=4))
def rect(b, region, x0, y0, s, color, lw, ls='-', z=5):
    k = s / (region[2] - region[0])
    X0, Y0 = x0 + (b[0] - region[0]) * k, y0 + (b[1] - region[1]) * k
    X1, Y1 = x0 + (b[2] - region[0]) * k, y0 + (b[3] - region[1]) * k
    X0, Y0, X1, Y1 = max(X0, x0), max(Y0, y0), min(X1, x0 + s), min(Y1, y0 + s)
    if X1 > X0 and Y1 > Y0:
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc='none', ec=color, lw=lw, ls=ls, zorder=z, joinstyle='miter'))
side = min(iw, ih); FULL = np.array([0, 0, side, side])

# ================================================================ panel a: inference pipeline
panel(1.0, 2.8, 'a'); T(5.0, 2.8, 'YOLO26-m with CellRefine at inference', size=FSH, weight='bold', ha='left')
YA = 8.0; S = 25.0                                   # top of the image row, image size
# input
header(4.0, 29.0, YA, 'Input', INK2)
draw_img(4.0, YA + 5.0, S, im.crop((0, 0, side, side)))
T(16.5, YA + S + 8.2, '640 × 640 × 3', size=FSS, color=INK2)
# YOLO26-m (frozen): existing modules, white with stage frames
header(34.0, 60.0, YA, 'YOLO26-m', INK2)
rbox(34.0, YA + 4.2, 60.0, YA + S + 5.8, fc='white', ec=MUTED, lw=0.5, ls=(0, (2.0, 1.2)), z=1)
for i, (nm, st) in enumerate((('Backbone', BB), ('Neck', NK), ('Head', HD))):
    y0 = YA + 6.0 + i * 8.6
    framed(36.0, y0, 58.0, y0 + 5.6, [nm], ec=st['ink'], color=st['ink'], size=FS)
    if i < 2: arrow((47.0, y0 + 5.6), (47.0, y0 + 8.6), color=st['ink'], lw=0.5)
T(47.0, YA + S + 8.2, 'trained once, then frozen', size=FSS, color=INK2)
arrow((29.2, YA + 5.0 + S / 2), (33.8, YA + 5.0 + S / 2))
# detections
header(65.0, 90.0, YA, 'Detections', INK2)
XD = 65.0
draw_img(XD, YA + 5.0, S, im.crop((0, 0, side, side)))
for c in (1, 0, 2):
    for b in b0[cl == c]:
        rect(b, FULL, XD, YA + 5.0, S, CLS[c], 0.55 if c == 1 else 0.75, ls=DASH)
arrow((60.2, YA + 5.0 + S / 2), (64.8, YA + 5.0 + S / 2))
T(XD + S / 2, YA + S + 8.2, 'boxes b, classes, scores', size=FSS, color=INK2)
# cell crops (one per class shown)
header(95.0, 112.0, YA, 'Cell crops', INK2)
SC = 7.6
for i, c in enumerate((0, 1, 2)):
    y0 = YA + 5.0 + i * (SC + 1.1)
    reg = ctx_box(b0[ex[c]])
    draw_img(98.6, y0, SC, crop(reg))
    rect(b0[ex[c]], reg, 98.6, y0, SC, CLS[c], 0.6, ls=(0, (1.6, 1.0)))
T(103.5, YA + S + 8.2, '1.6 × box, 128 × 128', size=FSS, color=INK2)
arrow((90.2, YA + 5.0 + S / 2), (97.4, YA + 5.0 + S / 2))
T(93.9, YA + 5.0 + S / 2 - 2.2, 'per box', size=5.2, color=INK2)
# CellRefine: proposed modules, filled
header(117.0, 143.0, YA, 'CellRefine', INK)
rbox(117.0, YA + 4.2, 143.0, YA + S + 5.8, fc='white', ec=INK, lw=0.6, z=1)
for i, (nm, st) in enumerate((('Crop encoder', BB), ('Fusion MLP', NK), ('Edge head', HD))):
    y0 = YA + 6.0 + i * 8.6
    filled(119.0, y0, 141.0, y0 + 5.6, [nm], st, size=FS)
    if i < 2: arrow((130.0, y0 + 5.6), (130.0, y0 + 8.6), color=st['ink'], lw=0.5)
arrow((107.4, YA + 5.0 + S / 2), (116.8, YA + 5.0 + S / 2))
# refined detections
header(148.0, 173.0, YA, 'Output', INK2)
XO = 148.0
draw_img(XO, YA + 5.0, S, im.crop((0, 0, side, side)))
for c in (1, 0, 2):
    for b in b1[cl == c]:
        rect(b, FULL, XO, YA + 5.0, S, CLS[c], 0.55 if c == 1 else 0.75)
T(XO + S / 2 - 2.0, YA + S + 8.2, 'refined boxes', size=FSS, color=INK2)
# box update formula and the skip connection of b
YS = YA + S + 14.4
line([(XD + S / 2, YA + S + 10.0), (XD + S / 2, YS)], color=INK2, lw=0.5)
line([(XD + S / 2, YS), (126.0, YS)], color=INK2, lw=0.5)
framed(126.0, YS - 3.4, 160.5, YS + 3.4, [r"$b' = b + \Delta \odot (w, h, w, h)$", 'class and score kept'], ec=INK2, fc=NEUT, size=FS, lh=2.9)
line([(160.5, YS), (171.6, YS), (171.6, YA + 5.0 + S + 0.3)], lw=0.5)
arrow((171.6, YA + 5.0 + S + 3.0), (171.6, YA + 5.0 + S + 0.2), lw=0.5)
arrow((130.0, YA + S + 5.8), (130.0, YS - 3.5), color=HD['ink'], lw=0.5)
T(XD + S / 2 + 1.2, YS - 1.4, r'$b$', size=FSS, color=INK2, ha='left')
T(131.2, YA + S + 8.6, r'$\Delta$', size=FS, color=HD['ink'], ha='left')
ax.plot([4.0, 173.0], [YS + 5.2, YS + 5.2], color=HAIR, lw=0.4)

# ================================================================ panel b: CellRefine network for one box
YB0 = YS + 7.8
panel(1.0, YB0, 'b'); T(5.0, YB0, 'CellRefine network for one detected box', size=FSH, weight='bold', ha='left')
YB = YB0 + 4.0; YM = YB + 13.0                        # top and middle line of panel b
reg = ctx_box(b0[ipl]); SB = 19.0
draw_img(4.0, YM - SB / 2, SB, crop(reg))
rect(b0[ipl], reg, 4.0, YM - SB / 2, SB, CLS[2], 0.8, ls=DASH)
T(13.5, YM + SB / 2 + 2.4, 'crop 128 × 128 × 3', size=FSS, color=INK2)
T(13.5, YM - SB / 2 - 2.0, 'detected platelet', size=FSS, color=INK2)
# encoder slabs (ResNet-18 stages)
names = [('stem', '32²×64', 9.5), ('layer1', '32²×64', 9.5), ('layer2', '16²×128', 7.6), ('layer3', '8²×256', 5.8), ('layer4', '4²×512', 4.2)]
x = 27.0
arrow((23.2, YM), (26.6, YM), color=INK2, lw=0.5)
for i, (n, d, sz) in enumerate(names):
    ax.add_patch(Rectangle((x, YM - sz / 2), 4.6, sz, fc=BB['ink'] if i else BB['mid'], ec=BB['ink'], lw=0.4, zorder=3))
    T(x + 2.3, YM + 7.6, n, size=5.2, color=BB['ink']); T(x + 2.3, YM + 9.8, d, size=5.0, color=INK2)
    if i < len(names) - 1: arrow((x + 4.6, YM), (x + 6.2, YM), color=BB['ink'], lw=0.45)
    x += 6.4
T(42.6, YM - 7.6, 'ResNet-18 encoder', size=FSS, color=BB['ink'], weight='bold')
filled(58.5, YM - 4.2, 72.0, YM + 4.2, ['1 × 1 conv', '128 ch, 2048-d'], BB, size=5.4, lh=2.4)
arrow((57.0, YM), (58.3, YM), color=BB['ink'], lw=0.45)
# fusion
cx = 77.0
ax.add_patch(Circle((cx, YM), 1.4, fc='white', ec=NK['ink'], lw=0.55, zorder=4)); T(cx, YM + 0.05, 'C', size=5.6, color=NK['ink'], weight='bold', zorder=5)
arrow((72.2, YM), (75.6, YM), color=BB['ink'], lw=0.5)
framed(68.8, YB - 2.4, 85.2, YB + 2.6, ['class one-hot (3)'], ec=INK2, size=5.2)
framed(68.8, YM + 8.0, 85.2, YM + 13.0, ['log w, log h (2)'], ec=INK2, size=5.2)
arrow((cx, YB + 2.6), (cx, YM - 1.4), color=INK2, lw=0.5); arrow((cx, YM + 8.0), (cx, YM + 1.4), color=INK2, lw=0.5)
filled(86.0, YM - 5.0, 104.0, YM + 5.0, ['MLP', '2053 → 512 → 512', 'ReLU, dropout 0.1'], NK, size=5.4, lh=2.4)
arrow((78.4, YM), (85.8, YM), color=NK['ink'], lw=0.5)
# edge head and expected offsets (simple linear chain)
filled(108.5, YM - 5.0, 125.5, YM + 5.0, ['Edge head', '4 × 17 logits'], HD, size=5.6, lh=2.6)
arrow((104.2, YM), (108.3, YM), color=NK['ink'], lw=0.5)
framed(130.0, YM - 5.0, 149.5, YM + 5.0, ['softmax per edge', r'$\Delta_j = \sum_k p_{j,k}\, t_k$'], ec=HD['ink'], fc=NEUT,
       color=INK, size=5.6, lh=3.0)
arrow((125.7, YM), (129.8, YM), color=HD['ink'], lw=0.5)
# refined box on the crop
SR = 19.0; XR = 153.5
draw_img(XR, YM - SR / 2, SR, crop(reg))
rect(b0[ipl], reg, XR, YM - SR / 2, SR, CLS[2], 0.8, ls=DASH)
rect(b1[ipl], reg, XR, YM - SR / 2, SR, CLS[2], 0.9)
arrow((149.7, YM), (153.3, YM), color=HD['ink'], lw=0.5)
T(XR + SR / 2, YM - SR / 2 - 2.0, 'refined platelet', size=FSS, color=INK2)
T(XR + SR / 2, YM + SR / 2 + 2.4, f'IoU {iou0:.2f} → {iou1:.2f}', size=FSS, color=INK2)
ax.plot([4.0, 173.0], [YM + SR / 2 + 5.4, YM + SR / 2 + 5.4], color=HAIR, lw=0.4)

# ================================================================ panel c: training
YC0 = YM + SR / 2 + 8.0
panel(1.0, YC0, 'c'); T(5.0, YC0, 'Training of CellRefine (YOLO26-m is not changed)', size=FSH, weight='bold', ha='left')
YC = YC0 + 3.4
framed(4.0, YC, 58.0, YC + 10.0, ['Input boxes on the training images', 'annotations jittered 6 times, σ ~ U(0.005, 0.15)',
       'YOLO26-m boxes, positive if IoU ≥ 0.3'], ec=INK2, size=5.6, lh=2.7)
framed(63.0, YC, 113.0, YC + 10.0, ['Targets', r"edge offsets $\hat{t} = (g - b) \oslash (w, h, w, h)$", 'IoU of b with its annotation g'],
       ec=INK2, size=5.6, lh=2.7)
filled(118.0, YC, 173.0, YC + 10.0, ['Loss', 'distribution focal loss + 5 × smooth L1', '+ binary cross-entropy of an auxiliary IoU head'], HD, size=5.6, lh=2.7)
arrow((58.2, YC + 5.0), (62.8, YC + 5.0)); arrow((113.2, YC + 5.0), (117.8, YC + 5.0))

for s in ('png', 'pdf'):
    fig.savefig(f'figures/output/Fig3_YOLO26m_CellRefine.{s}', dpi=600)
Image.open('figures/output/Fig3_YOLO26m_CellRefine.png').convert('RGB').save('figures/output/Fig3_YOLO26m_CellRefine.tiff', compression='tiff_lzw', dpi=(600, 600))
print('H used', YC + 10.0)
