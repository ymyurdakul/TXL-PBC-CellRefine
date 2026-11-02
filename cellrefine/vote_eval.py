"""Box voting with the one-to-many head of a single model.
mode e2e : YOLO26 default (one-to-one, NMS-free)
mode o2m : one-to-many + NMS
mode vote: one-to-many + NMS + voting (each kept box becomes the score^p weighted mean of candidates of the same
           class with IoU > vt)
"""
import argparse
import sys

import torch
import torchvision
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellrefine.common import DETECTORS, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--det", default="yolo26m_s0")
ap.add_argument("--weights", default="")
ap.add_argument("--modes", default="e2e,o2m,vote")
ap.add_argument("--vt", default="0.6,0.7,0.8")
ap.add_argument("--p", default="1,2")
ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--splits", default="val,test")
a = ap.parse_args()

CFG = {"vt": 0.7, "p": 1.0, "vote": False}


class VoteValidator(DetectionValidator):
    def postprocess(self, preds):
        if self.end2end:
            return super().postprocess(preds)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        out = []
        for x in preds:  # (4+nc, A)
            x = x.T.float()
            b = ops.xywh2xyxy(x[:, :4])
            sc, cl = x[:, 4:].max(1)
            m = sc > self.args.conf
            b, sc, cl = b[m], sc[m], cl[m]
            keep = torchvision.ops.batched_nms(b, sc, cl, self.args.iou)[: self.args.max_det]
            kb = b[keep]
            if CFG["vote"] and len(keep):
                M = torchvision.ops.box_iou(kb, b) * (cl[keep][:, None] == cl[None, :])
                W = (M > CFG["vt"]).float() * sc[None, :] ** CFG["p"]
                kb = (W @ b) / W.sum(1, keepdim=True).clamp(min=1e-9)
            out.append({"bboxes": kb, "conf": sc[keep], "cls": cl[keep].float(),
                        "extra": torch.zeros((len(keep), 0), device=kb.device)})
        return out


def run(mode, split, vt=0.7, p=1.0):
    CFG.update(vt=vt, p=p, vote=mode == "vote")
    m = YOLO(a.weights or str(DETECTORS[a.det]))
    if mode != "e2e":
        m.model.model[-1].end2end = False
    r = m.val(validator=VoteValidator, data=str(ROOT / "TXLPBC_runs" / "txlpbc.yaml"), split=split, imgsz=640,
              batch=16, device=0, iou=a.iou, plots=False, verbose=False, project=str(ROOT / "TXLPBC_runs/cellrefine/tmpval"),
              name="v", exist_ok=True)
    return r.box.map, r.box.map50, [r.box.class_result(i)[3] for i in range(3)]


for split in a.splits.split(","):
    for mode in a.modes.split(","):
        if mode == "vote":
            for vt in [float(v) for v in a.vt.split(",")]:
                for p in [float(v) for v in a.p.split(",")]:
                    mp, m50, c = run(mode, split, vt, p)
                    print(f"{a.det} {split} vote vt={vt} p={p}: mAP50-95={mp:.4f} mAP50={m50:.4f} cls={[round(float(v), 4) for v in c]}", flush=True)
        else:
            mp, m50, c = run(mode, split)
            print(f"{a.det} {split} {mode}: mAP50-95={mp:.4f} mAP50={m50:.4f} cls={[round(float(v), 4) for v in c]}", flush=True)
