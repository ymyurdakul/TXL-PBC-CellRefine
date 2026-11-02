# Aggregates CellRefine results from the raw metrics JSON files.
import json, glob, os, numpy as np, pandas as pd
M = 'TXLPBC_runs/cellrefine/metrics'
rows = []
for f in sorted(glob.glob(f'{M}/*.json')):
    j = json.load(open(f)); name = os.path.basename(f)[:-5]
    r = dict(file=name, det=j['det'], run=j['run'], tta=j.get('tta', 1), ep=j.get('ep'), params=j['refiner_params_M'],
             iters=j['chosen']['iters'], beta=j['chosen']['beta'], lam=j['chosen'].get('lam', 1.0),
             ms=j['refine_ms_per_img'], boxes=j['boxes_per_img'])
    for s in ('base_val', 'base_test', 'val', 'test', 'test_rescore_only', 'test_refine_only'):
        for k, v in j[s].items():
            r[f'{s}.{k}'] = v
    r['grid'] = j['val_grid']
    rows.append(r)
D = pd.DataFrame(rows).set_index('file')
D.to_pickle('cr_all.pkl')
cols = ['det', 'run', 'params', 'iters', 'lam', 'ms', 'boxes', 'base_val.mAP50-95', 'val.mAP50-95', 'base_test.mAP50-95', 'test.mAP50-95']
pd.set_option('display.width', 250)
print(D[cols].round(4).to_string())
# check: rescore-only == base and refine-only == final
print('rescore_only==base:', np.allclose(D['test_rescore_only.mAP50-95'], D['base_test.mAP50-95']), ' refine_only==test:', np.allclose(D['test_refine_only.mAP50-95'], D['test.mAP50-95']))
print('beta all zero:', (D.beta == 0).all())
