#!/bin/bash
# GPU 1: (a) apply the r18 refiner to yolo12n, (b) light resnet18_l3 refiner distilled from the r18_s0 teacher
cd "$(dirname "$0")/.."
L=TXLPBC_runs/cellrefine/logs
G="--iters 1,2 --betas 0 --lams 0.5,0.75,1.0"
export CUDA_VISIBLE_DEVICES=1
python cellrefine/eval_refine.py --det yolo12n --run r18_s0 $G --tag _on_yolo12n > $L/plug_r18_yolo12n.log 2>&1
python cellrefine/train_refine.py --det yolo26m_s0 --name l3kd_s0 --seed 0 --backbone resnet18_l3 --teacher r18_s0 > $L/l3kd_s0.log 2>&1
for D in yolo26m_s0 yolo12s yolo12n; do
  if [ $D = yolo26m_s0 ]; then T=_lam; else T=_on_$D; fi
  python cellrefine/eval_refine.py --det $D --run l3kd_s0 $G --tag $T > $L/ev_l3kd_s0$T.log 2>&1
done
echo DONE > $L/kd_done.txt
