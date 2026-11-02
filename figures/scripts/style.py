import matplotlib as mpl, numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
mpl.rcParams.update({
 'font.family':'Liberation Sans','font.size':6.5,'axes.linewidth':0.5,'axes.edgecolor':'#52514e',
 'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2,'ytick.major.size':2,
 'xtick.minor.width':0.4,'ytick.minor.width':0.4,'xtick.minor.size':1.2,'ytick.minor.size':1.2,
 'xtick.color':'#52514e','ytick.color':'#52514e','xtick.labelsize':6,'ytick.labelsize':6,
 'axes.labelcolor':'#1f1f1d','axes.labelsize':6.5,'text.color':'#1f1f1d',
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600,'figure.dpi':150,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'legend.fontsize':6,
 'lines.solid_capstyle':'round'})
INK='#1f1f1d'; INK2='#52514e'; MUTED='#8a8985'; GRID='#e7e6e2'; HAIR='#c9c8c3'
CLS=['WBC','RBC','Platelets']
COL={'WBC':'#0050FF','RBC':'#FF3300','Platelets':'#00D084'}  # paper palette (vivid, CVD-checked)
NAME={'WBC':'Leukocyte','RBC':'Erythrocyte','Platelets':'Thrombocyte'}
SHORT={'WBC':'WBC','RBC':'RBC','Platelets':'Platelets'}
SRC=['BCCD','BCDD','PBC','Raabin-WBC']
MM=1/25.4
def ramp(c):
    r=np.array(to_rgb(c)); dark=r*0.45
    return LinearSegmentedColormap.from_list('r',[(0,'#ffffff'),(0.55,r),(1,dark)])
class Page:
    def __init__(self,fig,W,H): self.f=fig; self.W=W; self.H=H
    def ax(self,x,y,w,h,**k):  # mm from top-left
        return self.f.add_axes([x/self.W,1-(y+h)/self.H,w/self.W,h/self.H],**k)
    def text(self,x,y,s,**k):
        return self.f.text(x/self.W,1-y/self.H,s,**k)
