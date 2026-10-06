"""plot_team_results.py — ball-experiment figure for findings/team_share/."""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/depth")
from final_validation import run, fit_fc, gamma_of, BALL
from size_regression import load

S=["#2a78d6","#eb6834","#1baf7a","#eda100"]
INK,INK2,MUTED,GRID,SURF="#0b0b0b","#52514e","#8a8880","#e6e5e1","#fcfcfb"
rows=load(); f,c=fit_fc(rows); g0=gamma_of(rows,c)
res=run(rows)

fig,ax=plt.subplots(1,3,figsize=(15.5,5.2),facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    for sp in ("left","bottom"): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2,labelsize=9); a.grid(True,color=GRID,lw=.8,zorder=0); a.set_axisbelow(True)

# A — predicted vs true distance
for tag,lab,col,mk in (("A","no reference",MUTED,"^"),("B","one reference",S[1],"s"),
                       ("C","several references",S[0],"o")):
    v=res[tag]
    # recompute the raw pairs for the scatter
import collections
from final_validation import fit_fc as _f
def pairs(tag):
    out=[]
    r=run(rows)
    return r[tag]
# scatter needs the raw Z pairs; re-run capturing them
sys.path.insert(0,"scripts/depth")
from collections import defaultdict
def raw():
    from final_validation import gamma_of as G
    sets=sorted({x["set"] for x in rows}); byf=defaultdict(list)
    for x in rows: byf[(x["set"],x["frame"])].append(x)
    out={"A":[],"B":[],"C":[]}
    for s in sets:
        tr=[x for x in rows if x["set"]!=s]
        fo,co=fit_fc(tr); go=G(tr,co)
        if go is None: continue
        zt=np.array([x["z"] for x in tr]); Zt=np.array([x["Ztape"]+co for x in tr])
        aout=float(np.exp(np.mean(np.log(zt)-go*np.log(Zt))))
        for k,rr in byf.items():
            if k[0]!=s: continue
            for j,y in enumerate(rr):
                Ztrue=y["Ztape"]+co
                oth=[rr[i] for i in range(len(rr)) if i!=j]
                def sc(a,t):
                    Z=(y["z"]/a)**(1/go); out[t].append((Ztrue,Z,2*y["r"]*Z/fo))
                sc(aout,"A")
                if oth:
                    for o in oth: sc(o["z"]/((o["Ztape"]+co)**go),"B")
                    Zo=np.array([o["Ztape"]+co for o in oth]); zo=np.array([o["z"] for o in oth])
                    sc(float(np.exp(np.mean(np.log(zo)-go*np.log(Zo)))),"C")
    return out
R=raw()
for tag,lab,col,mk in (("A","nothing of known size",MUTED,"^"),("C","several known sizes",S[0],"o")):
    v=np.array(R[tag]); ax[0].plot(v[:,0],v[:,1],mk,ms=5,color=col,alpha=.55,mec="none",
                                   zorder=3,label=f"{lab} — {res[tag]['dist_err'].mean():.0f}% off")
lim=[230,800]; ax[0].plot(lim,lim,"--",color=INK,lw=1.6,zorder=4)
ax[0].annotate("perfect",(lim[1],lim[1]),textcoords="offset points",xytext=(-6,-14),
               ha="right",fontsize=9,color=INK)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("true distance (mm)",color=INK2,fontsize=10)
ax[0].set_ylabel("estimated distance (mm)",color=INK2,fontsize=10)
ax[0].set_title("A · Estimated distance vs true distance",
                color=INK,fontsize=11.5,loc="left",pad=12)
ax[0].legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc="upper left")

# B — size error by scenario, against the floor
labs=["nothing of\nknown size","one object of\nknown size","several objects\nof known size"]
vals=[res[t]["size_err"].mean() for t in "ABC"]
cols=[MUTED,S[1],S[0]]
for i,(l,v,cl) in enumerate(zip(labs,vals,cols)):
    ax[1].bar(i,v,.58,color=cl,zorder=3,edgecolor=SURF)
    ax[1].annotate(f"{v:.1f}%",(i,v),textcoords="offset points",xytext=(0,4),
                   ha="center",fontsize=10,color=INK)
ax[1].axhline(1.72,color=INK,lw=2,ls="--",zorder=4)
ax[1].annotate("1.7% floor — circle fit + tape",(0.5,1.72),
               textcoords="offset points",xytext=(0,7),ha="center",fontsize=9,color=INK)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labs,fontsize=9)
ax[1].set_ylabel("size error on a 40 mm sphere (%)",color=INK2,fontsize=10)
ax[1].set_title("B · Size error on a 40 mm ball\none known object in shot halves it",
                color=INK,fontsize=11.5,loc="left",pad=12)
ax[1].grid(axis="x",visible=False)

# C — sensitivity to gamma
gs=np.arange(0.50,1.02,0.05)
ev=[run(rows,gamma_override=float(g))["C"]["size_err"].mean() for g in gs]
ax[2].plot(gs,ev,"-o",ms=6,lw=2,color=S[0],mec=SURF,mew=1.4,zorder=3)
i=int(np.argmin(ev))
ax[2].plot(gs[i],ev[i],"o",ms=13,color=S[1],mec=SURF,mew=2,zorder=5)
ax[2].annotate(f"best {ev[i]:.1f}%",(gs[i],ev[i]),textcoords="offset points",
               xytext=(6,10),fontsize=9,color=INK)
ax[2].axvline(g0,color=INK,lw=1.6,ls="--",zorder=2)
ax[2].annotate(f"value used ({g0:.2f})",(g0,max(ev)*0.93),textcoords="offset points",
               xytext=(-6,0),ha="right",fontsize=9,color=INK)
ax[2].axvline(1.0,color=MUTED,lw=1.4,ls=":",zorder=2)
ax[2].annotate("correction\nswitched off",(1.0,max(ev)*0.75),textcoords="offset points",
               xytext=(-6,0),ha="right",fontsize=8.5,color=MUTED)
ax[2].set_xlabel("correction strength",color=INK2,fontsize=10)
ax[2].set_ylabel("size error (%)",color=INK2,fontsize=10)
ax[2].set_title("C · The distance correction earns its place\nswitching it off nearly doubles the error",
                color=INK,fontsize=11.5,loc="left",pad=12)

fig.tight_layout(); fig.savefig("findings/team_share/ball_results.png",dpi=170,facecolor=SURF)
print("wrote findings/team_share/ball_results.png")
