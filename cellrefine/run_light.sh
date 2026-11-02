#!/bin/bash
# Light CellRefine: ResNet-18 without layer4 / MobileNetV3 backbone + a small detector (yolo12s) to reduce the total parameters
cd "$(dirname "$0")/.."
L=TXLPBC_runs/cellrefine/logs
G="--iters 1,2 --betas 0 --lams 0.5,0.75,1.0"
export CUDA_VISIBLE_DEVICES=0
# 0) apply the existing r18_s0 refiner to yolo12s (reference)
python cellrefine/eval_refine.py --det yolo12s --run r18_s0 $G --tag _on_yolo12s > $L/plug_r18_yolo12s.log 2>&1
# 1) resnet18_l3 on yolo26m_s0 (direct comparison with r18)
python cellrefine/train_refine.py --det yolo26m_s0 --name l3_s0 --seed 0 --backbone resnet18_l3 > $L/l3_s0.log 2>&1
python cellrefine/eval_refine.py --det yolo26m_s0 --run l3_s0 $G --tag _lam > $L/ev_l3_s0_lam.log 2>&1
python cellrefine/eval_refine.py --det yolo12s --run l3_s0 $G --tag _on_yolo12s > $L/plug_l3_yolo12s.log 2>&1
# 2) resnet18_l3 trained on the predictions of yolo12s, three seeds
for s in 0 1 2; do
  if [ $s -eq 0 ]; then D=yolo12s; else D=yolo12s_s$s; fi
  python cellrefine/train_refine.py --det $D --name l3_12s_s$s --seed $s --backbone resnet18_l3 > $L/l3_12s_s$s.log 2>&1
  python cellrefine/eval_refine.py --det $D --run l3_12s_s$s $G --tag _lam > $L/ev_l3_12s_s${s}_lam.log 2>&1
done
# 3) MobileNetV3-Large backbone (yolo26m_s0)
python cellrefine/train_refine.py --det yolo26m_s0 --name mnv3_s0 --seed 0 --backbone mnv3l > $L/mnv3_s0.log 2>&1
python cellrefine/eval_refine.py --det yolo26m_s0 --run mnv3_s0 $G --tag _lam > $L/ev_mnv3_s0_lam.log 2>&1
python cellrefine/eval_refine.py --det yolo12s --run mnv3_s0 $G --tag _on_yolo12s > $L/plug_mnv3_yolo12s.log 2>&1
echo DONE > $L/light_done.txt
