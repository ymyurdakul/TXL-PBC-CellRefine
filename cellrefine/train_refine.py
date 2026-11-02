"""CellRefine training. Example:
CUDA_VISIBLE_DEVICES=1 python cellrefine/train_refine.py --det yolo26m_s0 --name r18_s0 --seed 0
"""
import argparse
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellrefine.common import OUT, box_iou, cached_preds, evaluate, image_list, load_gt
from cellrefine.refiner import MEAN, STD, CellRefiner, crops, refine_preds

ap = argparse.ArgumentParser()
ap.add_argument("--det", default="yolo26m_s0")
ap.add_argument("--name", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--ctx", type=float, default=1.6)
ap.add_argument("--bins", type=int, default=16)
ap.add_argument("--rng", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--ipb", type=int, default=8, help="images per step")
ap.add_argument("--k", type=int, default=6, help="jittered proposals per ground truth box")
ap.add_argument("--smax", type=float, default=0.15, help="upper bound of the jitter SD (relative to box size)")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--w_l1", type=float, default=5.0)
ap.add_argument("--w_iou", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=4)
ap.add_argument("--geo", type=int, default=1, help="1: transpose and flip augmentation, 0: none")
ap.add_argument("--teacher", default="", help="run name of the teacher refiner for distillation (empty: none)")
ap.add_argument("--w_kd", type=float, default=1.0)
ap.add_argument("--kd_T", type=float, default=2.0)
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)
dev = "cuda"
RUN = OUT / "runs" / a.name
RUN.mkdir(parents=True, exist_ok=True)
(RUN / "args.json").write_text(json.dumps(vars(a), indent=2))


def load_split(split):
    ims, gts = {}, {}
    for p in image_list(split):
        im = np.array(Image.open(p).convert("RGB"))
        ims[p] = im
        gts[p] = load_gt(p, (im.shape[1], im.shape[0]))
    return ims, gts


t0 = time.time()
tr_im, tr_gt = load_split("train")
va_im, va_gt = load_split("val")
tr_det = cached_preds(a.det, "train")
va_det = cached_preds(a.det, "val")
base_val = evaluate(va_det, va_gt)
print(f"data ready {time.time() - t0:.0f}s | train {len(tr_im)} val {len(va_im)} | detector val mAP50-95="
      f"{base_val['mAP50-95']:.4f}", flush=True)

keys = list(tr_im)
gpu_im = {k: torch.from_numpy(v).to(dev).permute(2, 0, 1).contiguous() for k, v in tr_im.items()}
mean, std = MEAN.to(dev)[0], STD.to(dev)[0]


def aug_image(k):
    """D4 (transpose + flips) and color jitter; ground truth and detector boxes get the same transform."""
    img = gpu_im[k].float() / 255.0
    gb, gc = tr_gt[k]
    gb = torch.from_numpy(gb).to(dev)
    gc = torch.from_numpy(gc).to(dev)
    d = tr_det[k]
    db = torch.from_numpy(d["box"]).to(dev)
    dc = torch.from_numpy(d["cls"]).to(dev)
    boxes = [gb, db]
    g_on = a.geo == 1
    if g_on and random.random() < 0.5:  # transpose
        img = img.transpose(1, 2)
        boxes = [b[:, [1, 0, 3, 2]] for b in boxes]
    H, W = img.shape[1:]
    if g_on and random.random() < 0.5:
        img = img.flip(2)
        boxes = [torch.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1) for b in boxes]
    if g_on and random.random() < 0.5:
        img = img.flip(1)
        boxes = [torch.stack([b[:, 0], H - b[:, 3], b[:, 2], H - b[:, 1]], 1) for b in boxes]
    # color
    img = img * random.uniform(0.8, 1.2)
    m = img.mean()
    img = (img - m) * random.uniform(0.8, 1.2) + m
    g = img.mean(0, keepdim=True)
    img = (img - g) * random.uniform(0.7, 1.3) + g
    img = img.clamp(0, 1)
    img = (img - mean) / std
    return img, boxes[0], gc, boxes[1], dc


def proposals(gb, gc, db, dc):
    """Jittered ground truth proposals + detector predictions. Returns: box, class, target box, positive mask."""
    n = len(gb)
    P, C, T, M = [], [], [], []
    if n:
        wh = (gb[:, 2:] - gb[:, :2]).clamp(min=1.0)
        s = torch.empty(n, a.k, 1, device=dev).uniform_(0.005, a.smax)
        noise = torch.randn(n, a.k, 4, device=dev) * s * torch.cat([wh, wh], 1)[:, None]
        jb = (gb[:, None] + noise).view(-1, 4)
        jb = torch.cat([torch.minimum(jb[:, :2], jb[:, 2:] - 2), jb[:, 2:]], 1)
        P.append(jb)
        C.append(gc.repeat_interleave(a.k))
        T.append(gb.repeat_interleave(a.k, 0))
        M.append(torch.ones(len(jb), dtype=torch.bool, device=dev))
    if len(db):
        if n:
            iou = torch.from_numpy(box_iou(db.cpu().numpy(), gb.cpu().numpy())).to(dev)
            iou = iou * (dc[:, None] == gc[None, :])
            best, j = iou.max(1)
        else:
            best = torch.zeros(len(db), device=dev)
            j = torch.zeros(len(db), dtype=torch.long, device=dev)
        pos = best >= 0.3
        keep = pos | (torch.rand(len(db), device=dev) < 0.3)  # keep part of the negatives
        P.append(db[keep])
        C.append(dc[keep])
        T.append(gb[j[keep]] if n else db[keep])
        M.append(pos[keep])
    return torch.cat(P), torch.cat(C), torch.cat(T), torch.cat(M)


def iou_pair(a_, b_):
    lt = torch.maximum(a_[:, :2], b_[:, :2])
    rb = torch.minimum(a_[:, 2:], b_[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    ar = (a_[:, 2:] - a_[:, :2]).clamp(min=0).prod(1)
    br = (b_[:, 2:] - b_[:, :2]).clamp(min=0).prod(1)
    return inter / (ar + br - inter + 1e-7)


model = CellRefiner(size=a.size, ctx=a.ctx, bins=a.bins, rng=a.rng, backbone=a.backbone).to(dev)
teacher = None
if a.teacher:
    _ck = torch.load(OUT / "runs" / a.teacher / "best.pt", map_location="cpu", weights_only=False)
    _ar = _ck["args"]
    assert _ar["size"] == a.size and _ar["ctx"] == a.ctx and _ar["bins"] == a.bins and _ar["rng"] == a.rng
    teacher = CellRefiner(size=_ar["size"], ctx=_ar["ctx"], bins=_ar["bins"], rng=_ar["rng"], backbone=_ar["backbone"],
                          pretrained=False)
    teacher.load_state_dict(_ck["model"])
    teacher = teacher.to(dev).eval()
    for _p in teacher.parameters():
        _p.requires_grad_(False)
    print(f"teacher: {a.teacher} ({_ar['backbone']})", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps_ep = math.ceil(len(keys) / a.ipb)
total = steps_ep * a.epochs
warm = min(300, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
scaler = torch.amp.GradScaler()
print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | steps {total}", flush=True)

best, hist = -1, []
for ep in range(a.epochs):
    model.train()
    random.shuffle(keys)
    agg = np.zeros(4)
    for s in range(steps_ep):
        X, CL, WH, TD, TI, MS = [], [], [], [], [], []
        for k in keys[s * a.ipb:(s + 1) * a.ipb]:
            img, gb, gc, db, dc = aug_image(k)
            pb, pc, tb, m = proposals(gb, gc, db, dc)
            H, W = img.shape[1:]
            pb = torch.stack([pb[:, 0].clamp(-W * 0.1, W), pb[:, 1].clamp(-H * 0.1, H), pb[:, 2].clamp(0, W * 1.1),
                              pb[:, 3].clamp(0, H * 1.1)], 1)
            pb = torch.cat([pb[:, :2], torch.maximum(pb[:, 2:], pb[:, :2] + 2)], 1)
            wh = pb[:, 2:] - pb[:, :2]
            X.append(crops(img, pb, a.size, a.ctx))
            CL.append(pc)
            WH.append(wh)
            TD.append((tb - pb) / torch.cat([wh, wh], 1))
            TI.append(iou_pair(pb, tb) * m)
            MS.append(m)
        X, CL, WH, TD, TI, MS = map(torch.cat, (X, CL, WH, TD, TI, MS))
        with torch.autocast("cuda", dtype=torch.float16):
            delta, logits, iol = model(X, CL, WH)
        delta, logits, iol = delta.float(), logits.float(), iol.float()
        l_dfl = model.dfl_loss(logits[MS], TD[MS])
        l_l1 = F.smooth_l1_loss(delta[MS], TD[MS].clamp(-a.rng, a.rng), beta=0.02)
        l_iou = F.binary_cross_entropy_with_logits(iol, TI)
        loss = l_dfl + a.w_l1 * l_l1 + a.w_iou * l_iou
        if teacher is not None:  # knowledge distillation: edge distributions and IoU prediction of the larger refiner
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                _, t_lg, t_io = teacher(X, CL, WH)
            T = a.kd_T
            l_kd = F.kl_div(F.log_softmax(logits[MS] / T, -1), F.softmax(t_lg.float()[MS] / T, -1),
                            reduction="batchmean") * T * T / 4
            l_kd = l_kd + F.binary_cross_entropy_with_logits(iol, torch.sigmoid(t_io.float()))
            loss = loss + a.w_kd * l_kd
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        agg += [loss.item(), l_dfl.item(), l_l1.item(), l_iou.item()]
    agg /= steps_ep
    line = f"ep {ep + 1}/{a.epochs} loss {agg[0]:.4f} dfl {agg[1]:.4f} l1 {agg[2]:.5f} iou {agg[3]:.4f} | {time.time() - t0:.0f}s"
    if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
        res = {}
        for it, beta in ((0, 0.5), (1, 0.0), (1, 0.5)):
            r = evaluate(refine_preds(model, va_det, va_im, iters=it, beta=beta), va_gt)
            res[f"i{it}_b{beta}"] = r["mAP50-95"]
        sc = max(res.values())
        line += " | val " + " ".join(f"{k}={v:.4f}" for k, v in res.items())
        hist.append({"ep": ep + 1, **res})
        if sc > best:
            best = sc
            torch.save({"model": model.state_dict(), "args": vars(a), "ep": ep + 1, "val": res}, RUN / "best.pt")
            line += " *"
    print(line, flush=True)
torch.save({"model": model.state_dict(), "args": vars(a), "ep": a.epochs}, RUN / "last.pt")
(RUN / "hist.json").write_text(json.dumps({"base_val": base_val, "hist": hist}, indent=2))
print(f"done: best val {best:.4f} (detector {base_val['mAP50-95']:.4f})", flush=True)
