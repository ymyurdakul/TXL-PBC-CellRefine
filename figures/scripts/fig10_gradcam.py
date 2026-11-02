# Fig. 10 - Gradient-weighted class activation maps (HiResCAM variant of Grad-CAM) of the CellRefine edge offsets.
# For each example crop: detector box (dashed) and refined box (solid), then one map per edge offset.
import sys, pickle, numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from cellrefine.refiner import CellRefiner, crops, to_tensor, expand

plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
                     'lines.solid_capstyle': 'butt'})
CLS = {0: '#0050FF', 1: '#FF3300', 2: '#00D084'}
CN = {0: 'White blood cell', 1: 'Red blood cell', 2: 'Platelet'}
HD = dict(ink='#A3245F', mid='#DE8FB4')
INK, INK2, HAIR = '#1f1f1d', '#52514e', '#c9c8c3'
MM = 1 / 25.4; DASH = (0, (2.4, 1.4)); LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CMAP = LinearSegmentedColormap.from_list('hd', [(0, (1, 1, 1, 0)), (0.35, (0.871, 0.561, 0.706, 0.45)), (1, (0.639, 0.141, 0.373, 0.85))])

ck = torch.load('TXLPBC_runs/cellrefine/runs/r18_s0/best.pt', map_location='cpu', weights_only=False)
a = ck['args']; M = CellRefiner(size=a['size'], ctx=a['ctx'], bins=a['bins'], rng=a['rng'], backbone=a['backbone'], pretrained=False)
M.load_state_dict(ck['model']); M.eval()
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
PAIRS = pd.read_csv('results/analysis/pairs.csv'); PAIRS = PAIRS[PAIRS.seed == 0]
IMGS = 'TXL-PBC_Dataset-master/TXL-PBC/images/test/'
EX = [('bac60b0b98055c6ca9ab5ec0b6e4ec3a', 25), ('6d1b1d2ceb2653108cae15b2111bb9bc', 2),
      ('940b907e4df95624a2ab11407a289ebb', 10), ('85ed20f54a4b504fabb0fd4d67300fae', 5)]

def maps(stem, gi):
    key = [k for k in R[(0, 'test')]['base'] if stem in k][0]
    p0, p1 = R[(0, 'test')]['base'][key], R[(0, 'test')]['ref'][key]
    keep = p0['conf'] >= 0.25
    pr = PAIRS[(PAIRS.img == stem) & (PAIRS.gi == gi)].iloc[0]; pi = int(pr.pi)
    b = torch.tensor(p0['box'][keep][pi:pi + 1]); c = torch.tensor(p0['cls'][keep][pi:pi + 1]); b1 = p1['box'][keep][pi]
    im = np.array(Image.open(IMGS + stem + '.png').convert('RGB'))
    x = crops(to_tensor(im, 'cpu'), b, 128, 1.6)
    acts = {}
    h = M.body[LAYER].register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
    d, _, _ = M(x, c, b[:, 2:] - b[:, :2]); h.remove()
    A = acts['a']; cams = []
    for j in range(4):
        g = torch.autograd.grad(d[0, j], A, retain_graph=True)[0]
        cam = (g * A).sum(1, keepdim=True).abs()
        cam = F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False)[0, 0].detach().numpy()
        cams.append(cam / (cam.max() + 1e-12))
    # display crop (un-normalized, same region as the network input)
    reg = expand(b, 1.6)[0].numpy()
    o = int(max(im.shape[:2]))  # pad with the ImageNet mean colour, which the network receives outside the image
    canvas = Image.new('RGB', (im.shape[1] + 2 * o, im.shape[0] + 2 * o), (124, 116, 104)); canvas.paste(Image.fromarray(im), (o, o))
    crop_img = canvas.transform((512, 512), Image.EXTENT, tuple(reg + o), Image.BICUBIC)
    return crop_img, reg, b[0].numpy(), b1, int(c[0]), cams, pr

W = 174.0; TS = 27.0; GAP = 2.4; X0 = 25.0; TOP = 8.0; PITCH = TS + 3.6
NR = len(EX) + 1
H = TOP + NR * PITCH + 9.0
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
def T(x, y, s, size=6.2, color=INK, ha='center', va='center', weight='normal', **k):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **k)
def rect(bb, reg, x0, y0, color, lw, ls='-', z=6):
    k = TS / (reg[2] - reg[0])
    X0, Y0, X1, Y1 = x0 + (bb[0] - reg[0]) * k, y0 + (bb[1] - reg[1]) * k, x0 + (bb[2] - reg[0]) * k, y0 + (bb[3] - reg[1]) * k
    X0, Y0, X1, Y1 = max(X0, x0), max(Y0, y0), min(X1, x0 + TS), min(Y1, y0 + TS)
    ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc='none', ec=color, lw=lw, ls=ls, zorder=z, joinstyle='miter'))
    return X0, Y0, X1, Y1
heads = ['Crop', 'Left edge', 'Top edge', 'Right edge', 'Bottom edge']
for i, s in enumerate(heads):
    T(X0 + i * (TS + GAP) + TS / 2, TOP - 3.2, s, size=7.0, weight='bold')
for r, (stem, gi) in enumerate(EX):
    y0 = TOP + r * PITCH
    crop_img, reg, b0, b1, c, cams, pr = maps(stem, gi)
    T(X0 - 2.0, y0 + TS / 2 - 4.2, CN[c], size=6.2, ha='right', weight='bold', color=CLS[c])
    T(X0 - 2.0, y0 + TS / 2 - 1.2, pr.src, size=6.0, ha='right', color=INK2)
    T(X0 - 2.0, y0 + TS / 2 + 1.8, f'IoU {pr.iou0:.2f} → {pr.iou1:.2f}', size=6.0, ha='right', color=INK2)
    gray = np.asarray(crop_img.convert('L').convert('RGB'))
    for col in range(5):
        x0 = X0 + col * (TS + GAP)
        if col == 0:
            ax.imshow(np.asarray(crop_img), extent=(x0, x0 + TS, y0 + TS, y0), zorder=3, interpolation='lanczos')
            rect(b0, reg, x0, y0, CLS[c], 0.9, ls=DASH); rect(b1, reg, x0, y0, CLS[c], 0.9)
        else:
            j = col - 1
            ax.imshow(gray, extent=(x0, x0 + TS, y0 + TS, y0), zorder=3, interpolation='lanczos', alpha=0.85)
            ax.imshow(CMAP(cams[j]), extent=(x0, x0 + TS, y0 + TS, y0), zorder=4, interpolation='bilinear')
            X_0, Y_0, X_1, Y_1 = rect(b0, reg, x0, y0, INK2, 0.5, ls=DASH, z=5)
            seg = [((X_0, Y_0), (X_0, Y_1)), ((X_0, Y_0), (X_1, Y_0)), ((X_1, Y_0), (X_1, Y_1)), ((X_0, Y_1), (X_1, Y_1))][j]
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=INK, lw=1.3, zorder=6, solid_capstyle='butt')
        ax.add_patch(Rectangle((x0, y0), TS, TS, fc='none', ec=INK2, lw=0.4, zorder=7))
# mean maps over all test detections
MEAN = np.load('results/analysis/mean_maps.npy')
y0 = TOP + len(EX) * PITCH + 1.4
ax.plot([4.0, W - 2.0], [y0 - 2.2, y0 - 2.2], color=HAIR, lw=0.4)
T(X0 - 2.0, y0 + TS / 2 - 2.8, 'Mean of all', size=6.2, ha='right', weight='bold')
T(X0 - 2.0, y0 + TS / 2 + 0.2, '1,943 test', size=6.2, ha='right', weight='bold')
T(X0 - 2.0, y0 + TS / 2 + 3.2, 'detections', size=6.2, ha='right', weight='bold')
bx = np.array([TS * (1 - 1 / 1.6) / 2, TS * (1 + 1 / 1.6) / 2])
for col in range(5):
    x0 = X0 + col * (TS + GAP)
    ax.add_patch(Rectangle((x0, y0), TS, TS, fc='#f7f7f5', ec='none', zorder=2))
    X_0, X_1, Y_0, Y_1 = x0 + bx[0], x0 + bx[1], y0 + bx[0], y0 + bx[1]
    if col == 0:
        T(x0 + TS / 2, y0 + TS / 2, 'detector box at', size=5.6, color=INK2)
        T(x0 + TS / 2, y0 + TS / 2 + 2.6, 'the crop center', size=5.6, color=INK2)
    else:
        j = col - 1
        mm = torch.tensor(MEAN[j])[None, None]
        mm = F.interpolate(mm, size=(128, 128), mode='bilinear', align_corners=False)[0, 0].numpy()
        ax.imshow(CMAP(mm / mm.max()), extent=(x0, x0 + TS, y0 + TS, y0), zorder=4, interpolation='bilinear')
        seg = [((X_0, Y_0), (X_0, Y_1)), ((X_0, Y_0), (X_1, Y_0)), ((X_1, Y_0), (X_1, Y_1)), ((X_0, Y_1), (X_1, Y_1))][j]
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=INK, lw=1.3, zorder=6, solid_capstyle='butt')
    ax.add_patch(Rectangle((X_0, Y_0), X_1 - X_0, Y_1 - Y_0, fc='none', ec=INK2, lw=0.5, ls=DASH, zorder=5))
    ax.add_patch(Rectangle((x0, y0), TS, TS, fc='none', ec=INK2, lw=0.4, zorder=7))
# legend and colour scale
yl = y0 + TS + 3.4
hs = [Line2D([0], [0], color=INK2, lw=0.9, ls=DASH, label='YOLO26-m box'), Line2D([0], [0], color=INK2, lw=0.9, label='Refined box'),
      Line2D([0], [0], color=INK, lw=1.4, label='Edge explained by the map')]
fig.legend(handles=hs, loc='center', bbox_to_anchor=(0.36, 1 - yl / H), ncol=3, frameon=False, fontsize=6.0, handlelength=2.2, columnspacing=1.4)
cb = fig.add_axes([134.0 / W, 1 - (yl + 1.0) / H, 36 / W, 1.8 / H])
cb.imshow(CMAP(np.linspace(0, 1, 256))[None], aspect='auto', extent=(0, 1, 0, 1)); cb.set_yticks([])
cb.set_xticks([0, 1]); cb.set_xticklabels(['low', 'high'], fontsize=5.6); cb.tick_params(length=1.5, pad=1)
for s in cb.spines.values(): s.set_linewidth(0.4)
T(132.5, yl, 'Map value', size=6.0, ha='right')
out = 'figures/output/Fig10_gradcam' + ('' if LAYER == 5 else f'_L{LAYER}')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
Image.open(out + '.png').convert('RGB').save(out + '.tiff', compression='tiff_lzw', dpi=(600, 600))
print('H', H)
