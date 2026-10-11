"""KM 5모델 × 3조직 + 로그랭크 표. 유방 = TCGA-BRCA OS 단일 코호트 학습(CP-PCH · 4 베이스라인 모두 TCGA 860 학습 · 저장 환자 위험) · 외부 SCAN-B OS / METABRIC OS.
SIDISH 는 공식 배터리 3시드(v176 s0–s2)의 최종 반복(i4) 위험을 시드별 순위로 바꿔 평균한다 — 시드 0 단독은 METABRIC 에서 91% 가 같은 최소값(동률)이라 상위 20% 를 정의할 수 없었다."""
import os
import json,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def km(t,e):
    o=np.argsort(t); t,e=t[o],e[o]; n=len(t); S=1.; xs=[0.]; ys=[1.]
    for i,(ti,ei) in enumerate(zip(t,e)):
        if ei>0: S*=1-1/(n-i); xs.append(ti); ys.append(S)
    xs.append(t[-1] if len(t) else 0); ys.append(S); return np.array(xs),np.array(ys)
def logrank(t,e,g):
    ts=np.unique(t[e>0]); O1=E1=V=0.
    for x in ts:
        at=t>=x;n=at.sum();n1=(at&(g==1)).sum();d=((t==x)&(e>0)).sum();d1=((t==x)&(e>0)&(g==1)).sum()
        if n<2: continue
        O1+=d1;E1+=d*n1/n;V+=d*(n1/n)*(1-n1/n)*(n-d)/(n-1)
    from math import erfc,sqrt
    return float(erfc(abs((O1-E1)/np.sqrt(max(V,1e-12)))/sqrt(2))) if V>0 else np.nan
J=lambda p: json.load(open(p))
_ARM=J("results/report/BREAST_ARM.json"); BR_CP=J(os.environ.get("KM_BR_CP",f"results/w_exp/{_ARM['tag']}_{_ARM['suf']}_E2PP_RISKS.json"))   # 배포 유방 = TCGA OS 단일 코호트 헤드(시드 z 평균 위험)
BR_SC=J("results/report/BRCA_SCSURV_RISKS_s0.json"); BR_SD=[np.load(f"results/panel16k/SIDISH16K_RISK_v176s{s}.npz") for s in range(3)]   # (LEGACY860) SIDISH 공식 3시드 — 실제는 FAIR 731 적합판(R1 과 다름, 2026-09-26 감사)
BR_XT=J("results/report/BRCA_SCDG_RISKS.json")
LV_CP=J("results/report/LIVER_V5_RISKS.json"); LV_BL=J("results/report/LIVER_BL_RISKS_s0.json"); LV_XT=J("results/report/LIVER_SCDG_RISKS.json")
PC_CP=J("results/report/PANC_V5_RISKS.json"); PC_BL=J("results/report/PANC_BL_RISKS_s0.json"); PC_XT=J("results/report/PANC_SCDG_RISKS.json")
def gre(R,coh,ep):
    c=R[coh]; r=np.array(c["risk"],float); v=c["endpoints"][ep]
    t=np.array(v["time"],float); e=np.array(v["event"],float)
    idx=np.array(v.get("indices",list(range(len(t)))),int)
    ri=r[idx] if len(idx)==len(t) else r
    m=np.isfinite(ri)&np.isfinite(t)&np.isfinite(e)&(t>0)
    return ri[m],t[m],e[m]
def sid_b(coh,ep):
    key={"TCGA-heldout":"heldout","SCAN-B":"SCAN-B","METABRIC":"METABRIC"}[coh]
    from scipy.stats import rankdata
    rs=[np.array(z[f"s{s}_i4_{key}"],float) for s,z in enumerate(BR_SD)]   # 시드별 최종 반복 위험
    r=np.mean([rankdata(x)/len(x) for x in rs],0)   # 시드별 순위 평균 — 시드 0 은 METABRIC 에서 91% 동률(포화)이라 단독 사용 불가
    c=BR_CP[coh]; v=c["endpoints"][ep]
    t=np.array(v["time"],float); e=np.array(v["event"],float)
    if len(r)!=len(t):
        idx=np.array(v.get("indices",[]),int)
        if len(idx)==len(t) and len(r)>t.shape[0]: r=r[idx]
        else: return None
    m=np.isfinite(r)&np.isfinite(t)&np.isfinite(e)&(t>0)
    return r[m],t[m],e[m]
_BLT=os.environ.get("BL_TAG","ALL26_860")   # 유방 베이스라인: ALL26_860 = 26도너 참조 · 860 OS 재실행(네 방법 모두 같은 860 모델 · 시드0 대표) · LEGACY860 = 8/30 원 파일
if _BLT!="LEGACY860":
    BR_KM=J(f"results/report/BRCA_BL_KM_{_BLT}.json"); BR_SC=BR_KM["scSurv"]; BR_XT={"Scissor":BR_KM["Scissor"],"DEGAS":BR_KM["DEGAS"]}
    sid_b=lambda coh,ep: gre(BR_KM["SIDISH"],coh,ep)
CASES=[("Breast · SCAN-B OS (external)",{"CP-PCH":lambda: gre(BR_CP,"SCAN-B","OS"),"SIDISH":lambda: sid_b("SCAN-B","OS"),
        "scSurv":lambda: gre(BR_SC,"SCAN-B","OS"),"Scissor":lambda: gre(BR_XT["Scissor"],"SCAN-B","OS"),"DEGAS":lambda: gre(BR_XT["DEGAS"],"SCAN-B","OS")}),
       ("Breast · METABRIC OS (external)",{"CP-PCH":lambda: gre(BR_CP,"METABRIC","OS"),"SIDISH":lambda: sid_b("METABRIC","OS"),
        "scSurv":lambda: gre(BR_SC,"METABRIC","OS"),"Scissor":lambda: gre(BR_XT["Scissor"],"METABRIC","OS"),"DEGAS":lambda: gre(BR_XT["DEGAS"],"METABRIC","OS")}),
       ("Liver · GSE14520 OS",{"CP-PCH":lambda: gre(LV_CP,"GSE14520","OS"),"SIDISH":lambda: gre(LV_BL["sidish"],"GSE14520","OS"),
        "scSurv":lambda: gre(LV_BL["scsurv"],"GSE14520","OS"),"Scissor":lambda: gre(LV_XT["Scissor"],"GSE14520","OS"),
        "DEGAS":lambda: gre(LV_XT["DEGAS"],"GSE14520","OS")}),
       ("Pancreas · GSE62452 OS",{"CP-PCH":lambda: gre(PC_CP,"GSE62452","OS"),"SIDISH":lambda: gre(PC_BL["sidish"],"GSE62452","OS"),
        "scSurv":lambda: gre(PC_BL["scsurv"],"GSE62452","OS"),"Scissor":lambda: gre(PC_XT["Scissor"],"GSE62452","OS"),
        "DEGAS":lambda: gre(PC_XT["DEGAS"],"GSE62452","OS")})]
MODELS=["CP-PCH","SIDISH","scSurv","Scissor","DEGAS"]
fig,axes=plt.subplots(len(CASES),5,figsize=(18,3.2*len(CASES)))
tab={}; deg={}; nn={}
for i,(cs,fns) in enumerate(CASES):
    for j,m in enumerate(MODELS):
        ax=axes[i,j]; got=fns[m]()
        if got is None or len(got[0])<20:
            ax.axis("off"); tab[f"{cs}|{m}"]=float("nan"); continue
        ri,t,e=got; hi=ri>=np.quantile(ri,0.8); p=logrank(t,e,hi.astype(int)); tab[f"{cs}|{m}"]=p
        if m=="CP-PCH": nn[cs]={"n":int(len(t)),"events":int(e.sum())}
        if not np.isfinite(p) or hi.mean()>0.5:   # 위험이 상수에 가까워 상위 20% 를 정의할 수 없음(동률) → 미산출로 기록
            _u,_c=np.unique(ri,return_counts=True); deg[f"{cs}|{m}"]=float(_c.max()/len(ri)); tab[f"{cs}|{m}"]=float("nan"); ax.axis("off"); continue
        xs,ys=km(t,e); ax.step(xs/365.25,ys,where="post",color="#626c78",lw=1.3)
        xs,ys=km(t[hi],e[hi]); ax.step(xs/365.25,ys,where="post",color="#a33232",lw=1.7)
        ax.set_title(f"{m} (p={p:.1e})",fontsize=9.5); ax.set_ylim(0,1.02)
        if j==0: ax.set_ylabel(cs,fontsize=9.5)
        if i==len(CASES)-1: ax.set_xlabel("Years")
fig.suptitle("High-risk (top 20%) vs All — 5-model comparison (all organs TCGA-OS-trained; breast external SCAN-B OS / METABRIC OS, liver/pancreas external GSE cohorts)",y=0.995,fontsize=13)
fig.tight_layout(); fig.savefig("results/report/fig_km_5models_v6.png",dpi=135,bbox_inches="tight")
json.dump({**tab,"_degenerate":deg,"_n":nn},open("results/report/KM_5MODELS_LOGRANK_v6.json","w"),ensure_ascii=False,indent=1)
for k,v in tab.items(): print(f"{k}: {v:.2e}" if v==v else f"{k}: —",flush=True)
print("[saved] fig_km_5models_v6.png")
