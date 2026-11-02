# Mean HiResCAM map per edge offset over all test detections (score >= 0.25), refiner of run 0, second residual stage.
import sys, pickle, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from cellrefine.refiner import CellRefiner, crops, to_tensor
from PIL import Image
torch.set_num_threads(2)
ck = torch.load('TXLPBC_runs/cellrefine/runs/r18_s0/best.pt', map_location='cpu', weights_only=False)
a = ck['args']; m = CellRefiner(size=a['size'], ctx=a['ctx'], bins=a['bins'], rng=a['rng'], backbone=a['backbone'], pretrained=False)
m.load_state_dict(ck['model']); m.eval()
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb'))
IMGS = 'TXL-PBC_Dataset-master/TXL-PBC/images/test/'
S = np.zeros((4, 16, 16)); n = 0; SC = {c: np.zeros((4, 16, 16)) for c in (0, 1, 2)}; NC = {0: 0, 1: 0, 2: 0}
for key, p in R[(0, 'test')]['base'].items():
    keep = p['conf'] >= 0.25
    if keep.sum() == 0: continue
    B = torch.tensor(p['box'][keep]); C = torch.tensor(p['cls'][keep])
    im = np.array(Image.open(IMGS + key.split('/')[-1]).convert('RGB'))
    x = crops(to_tensor(im, 'cpu'), B, 128, 1.6)
    acts = {}
    h = m.body[5].register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
    d, _, _ = m(x, C, B[:, 2:] - B[:, :2]); h.remove()
    A = acts['a']; res = []
    for j in range(4):
        g = torch.autograd.grad(d[:, j].sum(), A, retain_graph=True)[0]
        res.append((g * A).sum(1).abs().detach().numpy())
    res = np.stack(res, 1); res = res / (res.sum((2, 3), keepdims=True) + 1e-12)
    S += res.sum(0); n += len(res)
    for c in (0, 1, 2):
        SC[c] += res[C.numpy() == c].sum(0); NC[c] += int((C.numpy() == c).sum())
np.save('results/analysis/mean_maps.npy', S / n)
np.save('results/analysis/mean_maps_cls.npy', np.stack([SC[c] / NC[c] for c in (0, 1, 2)]))
print('n', n, NC)
