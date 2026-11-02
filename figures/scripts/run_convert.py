# Runs the figure scripts without saving and converts their drawing axes to draw.io files.
import sys, types, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'figures/scripts'); sys.path.insert(0, 'analysis')
from mpl2drawio import convert
import matplotlib.figure as mf
OUT = 'figures/drawio'
import os; os.makedirs(OUT, exist_ok=True)
JOBS = [('analysis/fig2_general.py', 'Fig02_yolo_general_structure', 'Fig. 2'),
        ('analysis/fig3_system.py', 'Fig03_yolo26m_cellrefine', 'Fig. 3')]
orig_save = mf.Figure.savefig
mf.Figure.savefig = lambda self, *a, **k: None
from PIL import Image as _I
orig_open = _I.open
for src, name, title in JOBS:
    g = {'__name__': '__fig__', '__file__': src}
    code = open(src, encoding='utf8').read()
    code = code.replace("Image.open(out + '.png').convert('RGB').save(", "(lambda *a, **k: None)(")
    code = re.sub if False else code
    code = code.replace("Image.open('figures/output/Fig3_YOLO26m_CellRefine.png').convert('RGB').save(", "(lambda *a, **k: None)(")
    exec(compile(code, src, 'exec'), g)
    n = convert(g['fig'], g['ax'], f'{OUT}/{name}.drawio', g['W'], g['H'], title)
    print(name, n, 'cells')
    plt.close('all')
