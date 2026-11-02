# Aggregates all TXL-PBC baseline runs (45 models x 3 seeds) into tidy tables.
import json, glob, re, os
import numpy as np, pandas as pd

U = 'TXLPBC_runs'
OUT = 'results/benchmark'
NAMES = ['WBC', 'RBC', 'Platelets']

ORDER = [  # (ultralytics name, family, variant)
 ('yolov3u', 'YOLOv3', 'base'), ('yolov3-tinyu', 'YOLOv3', 'tiny'),
 ('yolov5nu', 'YOLOv5', 'n'), ('yolov5su', 'YOLOv5', 's'), ('yolov5mu', 'YOLOv5', 'm'), ('yolov5lu', 'YOLOv5', 'l'), ('yolov5xu', 'YOLOv5', 'x'),
 ('yolov6n', 'YOLOv6', 'n'), ('yolov6s', 'YOLOv6', 's'), ('yolov6m', 'YOLOv6', 'm'), ('yolov6l', 'YOLOv6', 'l'), ('yolov6x', 'YOLOv6', 'x'),
 ('yolov8n', 'YOLOv8', 'n'), ('yolov8s', 'YOLOv8', 's'), ('yolov8m', 'YOLOv8', 'm'), ('yolov8l', 'YOLOv8', 'l'), ('yolov8x', 'YOLOv8', 'x'),
 ('yolov9t', 'YOLOv9', 't'), ('yolov9s', 'YOLOv9', 's'), ('yolov9m', 'YOLOv9', 'm'), ('yolov9c', 'YOLOv9', 'c'), ('yolov9e', 'YOLOv9', 'e'),
 ('yolov10n', 'YOLOv10', 'n'), ('yolov10s', 'YOLOv10', 's'), ('yolov10m', 'YOLOv10', 'm'), ('yolov10b', 'YOLOv10', 'b'), ('yolov10l', 'YOLOv10', 'l'), ('yolov10x', 'YOLOv10', 'x'),
 ('yolo11n', 'YOLO11', 'n'), ('yolo11s', 'YOLO11', 's'), ('yolo11m', 'YOLO11', 'm'), ('yolo11l', 'YOLO11', 'l'), ('yolo11x', 'YOLO11', 'x'),
 ('yolo12n', 'YOLO12', 'n'), ('yolo12s', 'YOLO12', 's'), ('yolo12m', 'YOLO12', 'm'), ('yolo12l', 'YOLO12', 'l'), ('yolo12x', 'YOLO12', 'x'),
 ('yolo26n', 'YOLO26', 'n'), ('yolo26s', 'YOLO26', 's'), ('yolo26m', 'YOLO26', 'm'), ('yolo26l', 'YOLO26', 'l'), ('yolo26x', 'YOLO26', 'x'),
 ('rtdetr-l', 'RT-DETR', 'l'), ('rtdetr-x', 'RT-DETR', 'x'),
]
FAM = {m: f for m, f, v in ORDER}; VAR = {m: v for m, f, v in ORDER}; RANK = {m: i for i, (m, f, v) in enumerate(ORDER)}

def run_dir(model, seed):
    if seed == 0:
        return f'{U}/train/{model}'
    if model == 'yolo26m':
        return f'{U}/train_cell/base_yolo26m_s{seed}'
    return f'{U}/seeds/train/{model}_s{seed}'

def curve_stats(model, seed):
    d = pd.read_csv(run_dir(model, seed) + '/results.csv'); d.columns = d.columns.str.strip()
    m5095 = d['metrics/mAP50-95(B)'].fillna(0); m50 = d['metrics/mAP50(B)'].fillna(0)
    fit = 0.1 * m50 + 0.9 * m5095
    be = int(d.loc[fit.idxmax(), 'epoch'])
    loss_cols = [c for c in d.columns if c.startswith('train/')]
    nan_rows = d[d[loss_cols].isna().any(axis=1)]
    diverged_epoch = int(nan_rows['epoch'].min()) if len(nan_rows) and nan_rows['epoch'].min() > 5 else None
    return dict(epochs_run=len(d), best_epoch=be, best_val_map5095_curve=float(m5095.max()),
                last_val_map5095_curve=float(m5095.iloc[-1]), train_time_s=float(d['time'].iloc[-1]),
                s_per_epoch=float(d['time'].iloc[-1] / len(d)),
                final_train_loss=float(d[loss_cols].iloc[-1].sum()) if diverged_epoch is None else np.nan,
                diverged_epoch=diverged_epoch)

rows = []
for m, fam, var in ORDER:
    for seed in (0, 1, 2):
        f = f'{U}/metrics/{m}.json' if seed == 0 else f'{U}/metrics_seeds/{m}_s{seed}.json'
        j = json.load(open(f))
        cs = curve_stats(m, seed)
        for split in ('val', 'test'):
            r = dict(model=m, family=fam, variant=var, rank=RANK[m], seed=seed, split=split, source=j['source'],
                     pretrained=j['pretrained'], params_M=j['params_M'], GFLOPs=j['GFLOPs'],
                     train_time_h_json=j['train_time_h'], reused_from=j.get('reused_from', ''))
            r.update(cs)
            s = j[split]
            r.update(s)
            r['total_ms'] = s['preprocess_ms'] + s['inference_ms'] + s['postprocess_ms']
            r['FPS'] = 1000.0 / r['total_ms']
            r['FPS_inf'] = 1000.0 / s['inference_ms']
            r['F1'] = 2 * s['P'] * s['R'] / (s['P'] + s['R'])
            for n in NAMES:
                r[f'F1_{n}'] = 2 * s[f'P_{n}'] * s[f'R_{n}'] / (s[f'P_{n}'] + s[f'R_{n}'])
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/runs_long.csv', index=False)

METRICS = ['P', 'R', 'F1', 'mAP50', 'mAP75', 'mAP50-95'] + \
          [f'{k}_{n}' for n in NAMES for k in ('P', 'R', 'F1', 'AP50', 'AP50-95')] + \
          ['preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms', 'FPS', 'FPS_inf',
           'train_time_s', 's_per_epoch', 'best_epoch', 'best_val_map5095_curve', 'last_val_map5095_curve', 'final_train_loss']

def summarise(split):
    d = df[df.split == split]
    g = d.groupby('model')
    out = g[['family', 'variant', 'rank', 'params_M', 'GFLOPs', 'pretrained']].first()
    out['n_seeds'] = g['seed'].count()
    for k in METRICS:
        out[f'{k}_mean'] = g[k].mean(); out[f'{k}_sd'] = g[k].std(ddof=1)
        out[f'{k}_min'] = g[k].min(); out[f'{k}_max'] = g[k].max()
    out['diverged_seeds'] = g['diverged_epoch'].apply(lambda s: ','.join(str(int(x)) for x in s.dropna()) if s.notna().any() else '')
    out['diverged_runs'] = g['diverged_epoch'].apply(lambda s: int(s.notna().sum()))
    return out.sort_values('rank')

S = {sp: summarise(sp) for sp in ('test', 'val')}
for sp, t in S.items():
    t.to_csv(f'{OUT}/summary_{sp}.csv')
S['test'].to_json(f'{OUT}/summary_test.json', orient='index')
S['val'].to_json(f'{OUT}/summary_val.json', orient='index')

# ---- consistency checks against the author's own summary files
oz = pd.read_csv(f'{U}/ozet_tablo_test.csv')
chk = df[(df.split == 'test') & (df.seed == 0)].set_index('model')
bad = 0
for _, r in oz.iterrows():
    for a, b in (('P', 'P'), ('R', 'R'), ('mAP50', 'mAP50'), ('mAP50-95', 'mAP50-95'), ('Params (M)', 'params_M'), ('GFLOPs', 'GFLOPs'), ('AP50-95 Platelets', 'AP50-95_Platelets')):
        if abs(r[a] - chk.loc[r['Model'], b]) > 5e-5:
            bad += 1; print('MISMATCH', r['Model'], a, r[a], chk.loc[r['Model'], b])
print('ozet_tablo_test mismatches:', bad, 'of', len(oz) * 7)

# ---- quick report
t = S['test']
print(t[['family', 'variant', 'params_M', 'GFLOPs', 'mAP50_mean', 'mAP50_sd', 'mAP50-95_mean', 'mAP50-95_sd', 'AP50-95_Platelets_mean', 'FPS_mean', 'best_epoch_mean', 'diverged_seeds']].round(4).to_string())
print('divergence epochs:', df[df.diverged_epoch.notna()][['model', 'seed', 'diverged_epoch', 'best_epoch']].drop_duplicates().to_string())
print('train time total h:', df[df.split == 'test'].train_time_s.sum() / 3600)
