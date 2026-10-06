"""plot_trimmed_boll.py — figure for findings/trimmed_boll/README.md."""
import csv, json, glob, re, zipfile
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S=["#2a78d6","#eb6834","#1baf7a","#eda100"]
INK,INK2,MUTED,GRID,SURF="#0b0b0b","#52514e","#8a8880","#e6e5e1","#fcfcfb"
cal=json.load(open('findings/trimmed_boll/calibration.json'))
a_gt,g,f=cal['a_gt'],cal['gamma'],cal['focal_px']
pf=list(csv.DictReader(open('findings/trimmed_boll/per_frame.csv')))

z=zipfile.ZipFile("ground_truth/greenhouse_day_1/Cotton boll dimensions_Greenhouse 27Mar2026 (1).xlsx")
s=z.read("xl/worksheets/sheet1.xml").decode("utf8")
H=[]
for rn,body in re.findall(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>',s,re.S):
    if rn=="1": continue
    c={m.group(1):float(m.group(2)) for m in re.finditer(r'<c r="([A-Z]+)\d+"[^>]*>(?:<v>(.*?)</v>)?',body) if m.group(2)}
    if "C" in c: H.append(c["C"])
H=np.array(H)

def subset(minf,vis):
    cnt=defaultdict(int)
    rows=[r for r in pf if float(r['visfrac'])>=vis]
    for r in rows: cnt[r['track']]+=1
    rows=[r for r in rows if cnt[r['track']]>=minf]
    L=defaultdict(list); P=defaultdict(list)
    for r in rows:
        Z=(float(r['z'])/a_gt)**(1/g)
        L[r['track']].append(float(r['px'])*Z/f); P[r['track']].append(float(r['px']))
    return (np.array([np.median(v) for v in L.values()]),
            np.array([np.median(v) for v in P.values()]))

fig,ax=plt.subplots(1,3,figsize=(15.5,5.2),facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    for sp in ("left","bottom"): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2,labelsize=9); a.grid(True,color=GRID,lw=.8,zorder=0); a.set_axisbelow(True)

# A: what the trim did to DA3's depth range
def rng(pat,n=40):
    out=[]
    for p in sorted(glob.glob(pat))[:n]:
        d=np.load(p); d=d[np.isfinite(d)&(d>0)]
        out.append((np.percentile(d,1),np.percentile(d,99)))
    return np.mean([o[0] for o in out]),np.mean([o[1] for o in out])
sets=[("ball calibration\nscene",rng('frames/GX010117_ball*/depth/depth/*.npy'),S[2]),
      ("cotton, roof in frame\n(LeftDown_v5)",rng('inference_results/LeftDown_v5/depth/depth/*.npy'),S[1]),
      ("cotton, top 15% cropped\n(this run)",rng('inference_results/left_down_v5_trimmed/depth/depth/*.npy'),S[0])]
for i,(lab,(lo,hi),col) in enumerate(sets):
    ax[0].barh(i,hi-lo,left=lo,height=.45,color=col,zorder=3,edgecolor=SURF)
    ax[0].annotate(f"{lo:.2f} – {hi:.2f}",(hi,i),textcoords="offset points",xytext=(8,-3),fontsize=9,color=INK2)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels([s[0] for s in sets],fontsize=9)
ax[0].set_xscale("log"); ax[0].set_xlabel("DA3 raw depth, 1st–99th pct",color=INK2,fontsize=10)
ax[0].set_title("A · The roof was breaking the scale\ncropping it makes the scenes comparable",
                color=INK,fontsize=11.5,loc="left",pad=12)
ax[0].grid(axis="y",visible=False)

# B: spread vs ground truth
filts=[("all tracks",0,0.0),("≥10 frames",10,0.0),("≥10 fr, vis≥0.8",10,0.8),("≥20 fr, vis≥0.9",20,0.9)]
w=.36
for j,(lab,mf,vs) in enumerate(filts):
    L,P=subset(mf,vs)
    for k_,(v,col,off) in enumerate(((P,MUTED,-w/2),(L,S[0],w/2))):
        v=v/v.mean()*H.mean()
        ax[1].bar(j+off,100*v.std(ddof=1)/v.mean(),w*.9,color=col,zorder=3,edgecolor=SURF,
                  label=("pixels only" if j==0 and k_==0 else ("depth-corrected" if j==0 else None)))
        ax[1].annotate(f"{100*v.std(ddof=1)/v.mean():.0f}",(j+off,100*v.std(ddof=1)/v.mean()),
                       textcoords="offset points",xytext=(0,3),ha="center",fontsize=8,color=INK2)
ax[1].axhline(100*H.std(ddof=1)/H.mean(),color=INK,lw=2,zorder=4)
ax[1].annotate(f"ground truth {100*H.std(ddof=1)/H.mean():.1f}%",(3.4,100*H.std(ddof=1)/H.mean()),
               textcoords="offset points",xytext=(0,5),ha="right",fontsize=9,color=INK)
ax[1].set_xticks(range(len(filts))); ax[1].set_xticklabels([f[0] for f in filts],fontsize=8.5)
ax[1].set_ylabel("spread across bolls, CV (%)",color=INK2,fontsize=10)
ax[1].set_title("B · Depth beats pixels, and lands on the\ntrue spread once weak tracks are dropped",
                color=INK,fontsize=11.5,loc="left",pad=12)
ax[1].legend(frameon=False,fontsize=8.5,labelcolor=INK2); ax[1].grid(axis="x",visible=False)

# C: distribution comparison (ids are not matched, so compare quantiles)
L,P=subset(10,0.0); L=L/L.mean()*H.mean()
q=np.linspace(5,95,len(L))
ax[2].plot(np.percentile(H,q),np.sort(L),"o",ms=9,color=S[0],mec=SURF,mew=1.5,zorder=4,
           label=f"depth-corrected (n={len(L)})")
Pm=P/P.mean()*H.mean()
ax[2].plot(np.percentile(H,q),np.sort(Pm),"s",ms=8,color=MUTED,mec=SURF,mew=1.4,zorder=3,
           label=f"pixels only (n={len(Pm)})")
lim=[15,60]; ax[2].plot(lim,lim,"--",color=INK,lw=1.5,zorder=2)
ax[2].annotate("perfect agreement",(lim[1],lim[1]),textcoords="offset points",xytext=(-6,-14),
               ha="right",fontsize=9,color=INK2)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel("ground-truth boll height, quantile (mm)",color=INK2,fontsize=10)
ax[2].set_ylabel("measured length, quantile (mm)",color=INK2,fontsize=10)
ax[2].set_title("C · Distribution vs 38 measured bolls\nIDs unmatched, so quantiles are compared",
                color=INK,fontsize=11.5,loc="left",pad=12)
ax[2].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc="upper left")

fig.tight_layout(); fig.savefig("findings/trimmed_boll/trimmed_boll.png",dpi=170,facecolor=SURF)
print("wrote findings/trimmed_boll/trimmed_boll.png")
