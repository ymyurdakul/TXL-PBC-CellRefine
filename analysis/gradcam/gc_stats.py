# Side preference of HiResCAM maps of the four edge offsets (YOLO26-m run 0 refiner, all test detections with score >= 0.25)
import sys, pickle, json, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from cellrefine.refiner import CellRefiner, crops, to_tensor
from PIL import Image
torch.set_num_threads(2)
ck = torch.load('TXLPBC_runs/cellrefine/runs/r18_s0/best.pt', map_location='cpu', weights_only=False)
a = ck['args']; m = CellRefiner(size=a['size'], ctx=a['ctx'], bins=a['bins'], rng=a['rng'], backbone=a['backbone'], pretrained=False)
m.load_state_dict(ck['model']); m.eval()
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
IMGS = 'TXL-PBC_Dataset-master/TXL-PBC/images/test/'
out = {}
for layer, nm in ((5, 'layer2'), (6, 'layer3'), (7, 'layer4')):
    for mode in ('hires', 'gradcam'):
        M, CL = [], []
        for key, p in R[(0, 'test')]['base'].items():
            keep = p['conf'] >= 0.25
            if keep.sum() == 0: continue
            B = torch.tensor(p['box'][keep]); C = torch.tensor(p['cls'][keep])
            im = np.array(Image.open(IMGS + key.split('/')[-1]).convert('RGB'))
            x = crops(to_tensor(im, 'cpu'), B, 128, 1.6)
            acts = {}
            h = m.body[layer].register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
            d, _, u = m(x, C, B[:, 2:] - B[:, :2]); h.remove()
            A = acts['a']; res = []
            for j in range(4):
                g = torch.autograd.grad(d[:, j].sum(), A, retain_graph=True)[0]
                cam = (g * A).sum(1) if mode == 'hires' else (g.mean((2, 3), keepdim=True) * A).sum(1)
                res.append(cam.detach().numpy())
            res = np.abs(np.stack(res, 1)); n = res.shape[-1]; band = max(1, round(n * 0.35))
            tot = res.sum((2, 3)) + 1e-12
            L = res[..., :band].sum((2, 3)) / tot; Rr = res[..., -band:].sum((2, 3)) / tot
            T = res[..., :band, :].sum((2, 3)) / tot; Bt = res[..., -band:, :].sum((2, 3)) / tot
            M.append(np.stack([L[:, 0] - Rr[:, 0], T[:, 1] - Bt[:, 1], Rr[:, 2] - L[:, 2], Bt[:, 3] - T[:, 3]], 1)); CL.append(C.numpy())
        M = np.concatenate(M); CL = np.concatenate(CL)
        r = dict(n=int(len(M)), mean=M.mean(0).round(3).tolist(), frac_pos=(M > 0).mean(0).round(4).tolist(),
                 frac_pos_all=float((M > 0).mean()), per_class={int(c): float((M[CL == c] > 0).mean()) for c in (0, 1, 2)})
        out[f'{nm}_{mode}'] = r; print(nm, mode, r, flush=True)
json.dump(out, open('results/analysis/gc_stats.json', 'w'), indent=1)
