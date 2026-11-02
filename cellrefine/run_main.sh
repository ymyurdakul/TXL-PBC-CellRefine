#!/bin/bash
# CellRefine main experiment: three runs (detector seed k + refiner seed k)
cd "$(dirname "$0")/.."
L=TXLPBC_runs/cellrefine/logs
G="--iters 1,2 --betas 0 --lams 0.5,0.75,1.0"
export CUDA_VISIBLE_DEVICES=0
python cellrefine/eval_refine.py --det yolo26m_s0 --run r18_s0 $G --tag _lam > $L/ev_r18_s0_lam.log 2>&1
for s in 1 2; do
  python cellrefine/train_refine.py --det yolo26m_s$s --name r18_s$s --seed $s > $L/r18_s$s.log 2>&1
  python cellrefine/eval_refine.py --det yolo26m_s$s --run r18_s$s $G --tag _lam > $L/ev_r18_s${s}_lam.log 2>&1
done
echo DONE > $L/main_done.txt
