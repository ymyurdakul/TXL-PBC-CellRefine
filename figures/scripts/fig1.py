import sys; sys.path.insert(0,'figures/scripts')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter
from style import *
OUT=sys.argv[1]
B=pd.read_csv('boxes2.csv'); I=pd.read_csv('images.csv')
fields=np.load('fields_final.npy',allow_pickle=True).item()
protos={(a,b):(img,n) for a,b,img,n in np.load('protos.npy',allow_pickle=True)}
W,H=183,141
fig=plt.figure(figsize=(W*MM,H*MM),facecolor='white'); pg=Page(fig,W,H)
R_=fig.canvas.get_renderer()
def wmm(t): return t.get_window_extent(R_).width/fig.dpi*25.4
L=6; colw=(W-L-1)/3
nimg=B.groupby('cls').img.nunique(); ntot=len(B)
y_head=2.0; y_a=14.5; y_disk=17.5; dsz=55; y_b=y_disk+dsz+4.6; y_proto=y_b+5.6; tile=10.0
y_c=y_proto+tile+6.6; y_heat=y_c+3.6; hs=30
def row_title(y,letter,title):
    pg.text(1.0,y,letter,fontsize=9,fontweight='bold',va='baseline')
    pg.text(5.2,y,title,fontsize=6.8,color=INK2,va='baseline')
row_title(y_a,'a','Instance montage drawn at a common scale (640 × 640 network input)')
row_title(y_b,'b','Mean appearance per source dataset (instances normalised to box size, 1.3× context)')
row_title(y_c,'c','Spatial distribution of bounding-box centres')
for j,c in enumerate(CLS):
    x0=L+j*colw; cx=x0+colw/2
    n=(B.cls==c).sum()
    a=pg.ax(x0+1.8,y_head+0.9,3.0,3.0); a.add_patch(Circle((0.5,0.5),0.48,color=COL[c],clip_on=False)); a.set_xlim(0,1); a.set_ylim(0,1); a.axis('off')
    t=pg.text(x0+6.0,y_head+3.6,SHORT[c],fontsize=8.5,fontweight='bold',va='baseline')
    pg.text(x0+6.0+wmm(t)+1.4,y_head+3.6,f'{NAME[c].lower()}  ·  class {j}',fontsize=6.5,color=INK2,va='baseline')
    pg.text(x0+1.8,y_head+8.0,f'n = {n:,} ({100*n/ntot:.1f}%)  ·  present in {nimg[c]:,} of {len(I):,} images',fontsize=6.2,color=INK2,va='baseline')
    # (a) field
    canvas,R,g=fields[c]; S=canvas.shape[0]; m=R+16
    a=pg.ax(cx-dsz/2,y_disk,dsz,dsz)
    a.imshow(canvas,extent=[-S/2,S/2,S/2,-S/2],interpolation='antialiased')
    a.add_patch(Circle((0,0),R+8,fill=False,lw=0.5,ec=HAIR))
    a.set_xlim(-m,m); a.set_ylim(m,-m); a.set_aspect('equal'); a.axis('off')
    bx1=m-10; bx0=bx1-100; by=m-12
    a.plot([bx0,bx1],[by,by],color=INK,lw=1.0,solid_capstyle='butt')
    a.text((bx0+bx1)/2,by-14,'100 px',ha='center',va='bottom',fontsize=5.8,color=INK2)
    lab=f'{len(g)} shown' + (' (all eligible)' if c=='Platelets' else ' (random)')
    a.text(-m+4,by,lab,ha='left',va='center',fontsize=5.8,color=INK2)
    # (b) prototypes
    srcs=SRC+['All']; lbl=['BCCD','BCDD','PBC','Raabin','Pooled']
    gap=0.9; tw=tile; extra=1.4; totw=5*tw+4*gap+extra; px0=cx-totw/2
    for k,s in enumerate(srcs):
        img,nn=protos[(c,s)]
        xx=px0+k*(tw+gap)+(extra if k==4 else 0)
        a=pg.ax(xx,y_proto,tw,tw); a.imshow(np.clip(img,0,1),interpolation='bicubic'); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4 if k<4 else 0.6); sp.set_color(HAIR if k<4 else INK2)
        pg.text(xx+tw/2,y_proto-1.0,lbl[k],ha='center',va='baseline',fontsize=5.8,color=INK if k==4 else INK2,fontweight='bold' if k==4 else 'normal')
        pg.text(xx+tw/2,y_proto+tw+2.5,f'n = {nn:,}',ha='center',va='baseline',fontsize=5.4,color=MUTED)
    # (c) centre density
    g2=B[B.cls==c]; nb=64
    h2,_,_=np.histogram2d(g2.y,g2.x,bins=nb,range=[[0,1],[0,1]])
    h2=gaussian_filter(h2,2.5,mode='reflect'); h2=h2/h2.mean()
    vmax={'WBC':60,'RBC':2.2,'Platelets':2.6}[c]; ticks={'WBC':[0,1,5,20,40,60],'RBC':[0,0.5,1,1.5,2],'Platelets':[0,0.5,1,1.5,2,2.5]}[c]
    a=pg.ax(cx-hs/2-2.2,y_heat,hs,hs)
    nrm=PowerNorm(0.5,vmin=0,vmax=vmax) if c=='WBC' else None
    im=a.imshow(h2,extent=[0,1,1,0],cmap=ramp(COL[c]),norm=nrm,vmin=None if nrm else 0,vmax=None if nrm else vmax,interpolation='bicubic')
    a.set_xticks([0,.5,1]); a.set_yticks([0,.5,1]); a.set_xticklabels(['0','0.5','1']); a.set_yticklabels(['0','0.5','1'] if j==0 else [])
    for sp in ['top','right']: a.spines[sp].set_visible(True)
    for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color(INK2)
    a.set_xlabel('x / image width',labelpad=1.5)
    if j==0: a.set_ylabel('y / image height',labelpad=1.5)
    ca=pg.ax(cx+hs/2-0.6,y_heat,1.3,hs)
    cb=fig.colorbar(im,cax=ca); cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5.4,width=0.4,length=1.5,pad=1)
    cb.set_ticks(ticks); cb.set_ticklabels([f'{t:g}' for t in ticks])
    cb.set_label('× uniform density'+(' (√ scale)' if c=='WBC' else ''),fontsize=5.6,labelpad=2,color=INK2)
for ext in ['png','tiff','pdf']:
    kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
    fig.savefig(f'{OUT}.{ext}',dpi=600,facecolor='white',**kw)
print('ok')
