# AP of each class at each IoU threshold, with and without CellRefine (YOLO26-m, three runs, test split).
# Needs results/analysis/refined_all.pkl from analysis/reproduce_cellrefine.py. Writes results/analysis/ap_thr.pkl
# (arrays seed x class x threshold, in %) and a CSV of the mean gains.
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
import evalcore as E
R = pickle.load(open('results/analysis/refined_all.pkl', 'rb')); G = E.load_gt('test')
stem = lambda P: {os.path.basename(k)[:-4]: v for k, v in P.items()}
A0 = np.array([E.evaluate(stem(R[(s, 'test')]['base']), G)['ap'] for s in (0, 1, 2)]) * 100
A1 = np.array([E.evaluate(stem(R[(s, 'test')]['ref']), G)['ap'] for s in (0, 1, 2)]) * 100
pickle.dump(dict(A0=A0, A1=A1), open('results/analysis/ap_thr.pkl', 'wb'))
D = (A1 - A0).mean(0)
pd.DataFrame(D.T, index=np.round(np.linspace(0.5, 0.95, 10), 2), columns=['WBC', 'RBC', 'Platelet']).to_csv(
    'results/analysis/ap_gain_per_threshold.csv', float_format='%.3f')
print(np.round(D, 2))
