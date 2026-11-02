"""CellRefine: a light second stage that refines detector boxes on high-resolution cell crops and can rescore
them with a predicted IoU.

For each box, a context region (box x ctx) is cropped from the original image with RoIAlign at S x S. The network
predicts edge corrections (dx1, dy1, dx2, dy2, normalized by the box size) and the IoU of the box with its ground
truth box. Each edge is modeled as a discrete distribution (DFL style, K bins over -R..R); its expectation gives
the correction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def expand(boxes, ctx):
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0) * ctx / 2
    return torch.cat([c - wh, c + wh], 1)


def crops(img, boxes, size, ctx):
    """img: (3,H,W) float, normalized; boxes: (N,4) -> (N,3,S,S)"""
    rois = torch.cat([boxes.new_zeros(len(boxes), 1), expand(boxes, ctx)], 1)
    return roi_align(img[None], rois, (size, size), 1.0, 2, aligned=True)


class CellRefiner(nn.Module):
    def __init__(self, nc=3, size=128, ctx=1.6, bins=16, rng=0.5, backbone="resnet18", pretrained=True):
        super().__init__()
        self.nc, self.size, self.ctx, self.bins, self.rng = nc, size, ctx, bins, rng
        if backbone == "resnet18":
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            ch = 512
        elif backbone == "resnet34":
            m = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            ch = 512
        elif backbone == "resnet18_l3":  # light: layer4 removed (about 8.4M fewer parameters)
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            ch = 256
        elif backbone == "mnv3l":  # light: MobileNetV3-Large feature extractor
            m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
            ch = 960
        else:
            raise ValueError(backbone)
        if backbone.startswith("mnv3"):
            self.body = m.features
        elif backbone.endswith("_l3"):
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        else:
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        s = size // 32
        # AdaptiveAvgPool has no parameters; identity for resnet18 at 128 px, compatible with older weights
        self.red = nn.Sequential(nn.Conv2d(ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(s))
        d = 128 * s * s + nc + 2
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(512, 512),
                                 nn.ReLU(inplace=True))
        self.edge = nn.Linear(512, 4 * (bins + 1))
        self.iou = nn.Linear(512, 1)
        self.register_buffer("proj", torch.linspace(-rng, rng, bins + 1))

    def forward(self, x, cls, wh):
        f = self.red(self.body(x)).flatten(1)
        z = torch.cat([f, F.one_hot(cls, self.nc).float(), torch.log(wh.clamp(min=1.0)) / 6.0], 1)
        h = self.mlp(z)
        logits = self.edge(h).view(-1, 4, self.bins + 1)
        delta = (logits.softmax(-1) * self.proj).sum(-1)
        return delta, logits, self.iou(h).squeeze(1)

    def dfl_loss(self, logits, target):
        """target: (N,4) normalized edge corrections, clipped to [-rng, rng] and split between the two nearest bins."""
        t = (target.clamp(-self.rng, self.rng * 0.9999) + self.rng) / (2 * self.rng) * self.bins
        tl = t.floor().long()
        wr = t - tl
        lp = logits.log_softmax(-1)
        l = -(lp.gather(-1, tl[..., None]).squeeze(-1) * (1 - wr) + lp.gather(-1, (tl + 1)[..., None]).squeeze(-1) * wr)
        return l.mean()


def decode(boxes, delta):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0)
    return boxes + delta * torch.cat([wh, wh], 1)


def to_tensor(img_np, device):
    t = torch.from_numpy(img_np).to(device).permute(2, 0, 1).float() / 255.0
    return ((t[None] - MEAN.to(device)) / STD.to(device))[0]


def _tta_forward(model, x, cc, wh, tta):
    """D4 test-time augmentation: the crop is flipped/transposed, and the edge corrections are mapped back and averaged."""
    if tta <= 1:
        d, _, u = model(x, cc, wh)
        return d.float(), u.float().sigmoid()
    ds, us = [], []
    for t in range(tta):
        tr, hf, vf = (t >> 2) & 1, t & 1, (t >> 1) & 1
        xi, whi = x, wh
        if tr:
            xi, whi = xi.transpose(2, 3), wh.flip(1)
        if hf:
            xi = xi.flip(3)
        if vf:
            xi = xi.flip(2)
        d, _, u = model(xi, cc, whi)
        d = d.float()
        if vf:
            d = torch.stack([d[:, 0], -d[:, 3], d[:, 2], -d[:, 1]], 1)
        if hf:
            d = torch.stack([-d[:, 2], d[:, 1], -d[:, 0], d[:, 3]], 1)
        if tr:
            d = d[:, [1, 0, 3, 2]]
        ds.append(d)
        us.append(u.float().sigmoid())
    return torch.stack(ds).mean(0), torch.stack(us).mean(0)


@torch.no_grad()
def refine_preds(model, preds, images, iters=1, beta=0.5, chunk=512, device="cuda", tta=1, lam=1.0):
    """preds: {img: dict(box, conf, cls, wh)}; images: {img: HxWx3 uint8 RGB}
    beta: new score = conf^(1-beta) * iou^beta. iters: number of successive refinement passes. tta: 1, 4, or 8.
    lam: final box = lam * refined + (1 - lam) * detector box."""
    model.eval()
    out = {}
    for k, p in preds.items():
        q = dict(p)
        if len(p["box"]) == 0:
            out[k] = q
            continue
        img = to_tensor(images[k], device)
        b = torch.from_numpy(p["box"]).to(device)
        c = torch.from_numpy(p["cls"]).to(device)
        H, W = img.shape[1:]
        for it in range(iters + 1):  # refine `iters` times; the last pass only predicts the IoU
            nb, ious = [], []
            for i in range(0, len(b), chunk):
                bb, cc = b[i:i + chunk], c[i:i + chunk]
                x = crops(img, bb, model.size, model.ctx)
                with torch.autocast("cuda", dtype=torch.float16):
                    d, u = _tta_forward(model, x, cc, bb[:, 2:] - bb[:, :2], tta)
                nb.append(decode(bb, d))
                ious.append(u)
            iou = torch.cat(ious)
            if it < iters:
                b = torch.cat(nb)
                b[:, 0::2] = b[:, 0::2].clamp(0, W)
                b[:, 1::2] = b[:, 1::2].clamp(0, H)
        q["box"] = (lam * b.cpu().numpy() + (1 - lam) * p["box"]).astype(np.float32)
        q["iou"] = iou.cpu().numpy().astype(np.float32)
        q["conf"] = (p["conf"] ** (1 - beta) * q["iou"] ** beta).astype(np.float32)
        q["conf0"] = p["conf"]
        out[k] = q
    return out
