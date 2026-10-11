#!/usr/bin/env python3
"""B1 · Do the fitted hazard coefficients agree with established biology?

A named coordinate system makes the model's answer falsifiable: we can write down, BEFORE
looking at the fit, which direction each well-established program should push prognosis, and
then count how often the fitted W agrees.  The null is a full-training permutation: survival
labels are shuffled and Stage 3 is refit, giving the concordance achievable by chance.

Pre-registered exclusions (decided in the E2'' sign audit, not here): an axis is dropped if
|corr(M_ck, pi_tumor)| > 0.3, because composition adjustment can flip such coefficients.
ORG=liver|pancreas|breast  VARIANT=prime2
"""
from __future__ import annotations
import p16k_env  # 새 패널 주입 (다른 import 보다 먼저)
import json, os, importlib
import numpy as np, torch
torch.set_num_interop_threads=lambda n: None

ORG=os.environ.get("ORG","liver"); dev=torch.device(os.environ.get("DEV","cuda:0"))
R_NULL=int(os.environ.get("R_NULL","200"))

# ── pre-registered expectations: (cell-type key, axis, expected sign, one-line rationale) ──
EXPECT={
 "liver":[("Malignant cells","HCC_PROLIF_CLASS",+1,"Hoshida 증식형(S1/S2) = 불량 예후"),
          ("Malignant cells","HM_Hypoxia",+1,"HIF 활성 → 혈관신생·치료저항"),
          ("Malignant cells","HM_G2_M_Checkpoint",+1,"세포주기 진행 = 공격적 표현형"),
          ("Malignant cells","HM_E2F_Targets",+1,"E2F 표적 = 증식"),
          ("Malignant cells","HM_Bile_Acid_Metabolism",-1,"분화된 간세포 기능 보존"),
          ("Malignant cells","HM_Xenobiotic_Metabolism",-1,"간 해독 기능 = 분화 보존"),
          ("Malignant cells","HM_Fatty_Acid_Metabolism",-1,"성숙 간세포 대사"),
          ("TAMs","TAM_SPP1",+1,"SPP1+ TAM = 면역억제·섬유화"),
          ("TECs","HM_Angiogenesis",+1,"혈관신생 = 불량 예후")],
 "pancreas":[("Epithelial","PDAC_BASAL",+1,"Moffitt basal-like = 불량 예후"),
          ("Epithelial","PDAC_CLASSICAL",-1,"classical 아형 = 상대적 양호"),
          ("Epithelial","ACINAR",-1,"선방 분화 보존"),
          ("Epithelial","DUCTAL_ID",-1,"정상 도관 정체성 보존"),
          ("Epithelial","HM_G2_M_Checkpoint",+1,"증식"),
          ("Epithelial","HM_Hypoxia",+1,"저산소"),
          ("Immune","TREG",+1,"조절 T 세포 = 면역억제"),
          ("Fibroblast","CAF_MYO",+1,"myCAF 주도 섬유화")],
 "breast":[("Cancer Epithelial","ER_SIGNALING",-1,"ER 양성 = 양호 예후"),
          ("Cancer Epithelial","HM_Estrogen_Response_Early",-1,"에스트로겐 반응 = luminal"),
          ("Cancer Epithelial","HM_Estrogen_Response_Late",-1,"에스트로겐 반응 = luminal"),
          ("Cancer Epithelial","BASAL_PROGRAM",+1,"기저형 = 불량 예후"),
          ("Cancer Epithelial","HM_G2_M_Checkpoint",+1,"증식"),
          ("Cancer Epithelial","HM_Hypoxia",+1,"저산소"),
          ("Myeloid","TAM_SPP1",+1,"SPP1+ TAM = 면역억제"),
          ("Endothelial","HM_Angiogenesis",+1,"혈관신생")],
}

def setup_breast():
    import run_breast_e2_prime2 as P2; P2.install(dev)
    import run_breast_e2_prime as BP
    import run_breast_online_domain_adaptive_h as e2
    import run_breast_stage3_preprocess_v2 as protocol
    from evaluate_stage2_redesign import Full, fit_stage3
    bundle=torch.load(BP.REGISTRY,map_location="cpu",weights_only=False)
    data=torch.load(bundle["source"]["stage1_data"],map_location="cpu",weights_only=False)
    a=protocol.load_a(bundle,dev); mapping=protocol.fine_mapping(bundle)
    C=len(bundle["cell_type_names"])
    tr=np.asarray(bundle["bulk_split"]["train"]); x=bundle["bulk_x"][tr].float()
    His=[BP.load_h(bundle,BP.checkpoint_path("E2_prime",s),dev)[0] for s in BP.SEEDS]
    pis,sts=[],[]
    for h in His:
        with torch.no_grad():
            fi,st=[],[]
            for s in range(0,len(x),256):
                xb=x[s:s+256].to(dev); co,im=a(xb)
                fi.append(protocol.make_fine_pi(co,im,mapping,C).cpu()); st.append(h(xb,co.detach()).cpu())
            pis.append(torch.cat(fi)); sts.append(torch.cat(st))
    return bundle,data,tr,pis,sts,list(bundle["cell_type_names"]),BP.PROGRAMS,fit_stage3

def setup_organ():
    import run_organ_e2_prime2 as P2; P2.install(dev); E2P=P2.E2P
    L,LS3=E2P.L,E2P.LS3
    from evaluate_stage2_redesign import fit_stage3
    bundle=L.prepare_registry(); E2P.patch_cache(bundle); a=L.load_a(bundle,dev)
    data=torch.load(L.STAGE_DATA,map_location="cpu",weights_only=False)
    mapping=LS3.fine_mapping(bundle); tr=np.asarray(bundle["bulk_split"]["train"])
    pis,sts=[],[]
    for s in E2P.SEEDS:
        h=E2P.load_hp(bundle,s,dev)
        pi,mst,_=LS3.infer(a,h,bundle["bulk_x"][tr],mapping,bundle,dev)
        pis.append(pi.cpu()); sts.append(mst.cpu())
    return bundle,data,tr,pis,sts,list(bundle["cell_type_names"]),E2P.PROGRAMS,fit_stage3

bundle,data,tr,pis,sts,types,PROGRAMS,fit_stage3=(setup_breast() if ORG=="breast" else setup_organ())
time=torch.tensor(np.asarray(data["bulk_survival_time"])[tr],dtype=torch.float32,device=dev)
event=torch.tensor(np.asarray(data["bulk_survival_event"])[tr],dtype=torch.float32,device=dev)

def fitW(t,e):
    Ws=[]
    for pi,st in zip(pis,sts):
        Ws.append(fit_stage3(pi.to(dev),st.to(dev),t,e,dev,seed=0).W.detach().cpu())
    return torch.stack(Ws).mean(0)

W=fitW(time,event)
pi_mean=torch.stack(pis).mean(0); st_mean=torch.stack(sts).mean(0)
tum=types.index({"liver":"Malignant cells","pancreas":"Epithelial","breast":"Cancer Epithelial"}[ORG])
pit=pi_mean[:,tum].numpy()

rows=[]; keep=[]
for ct,ax,sign,why in EXPECT[ORG]:
    if ct not in types or ax not in PROGRAMS:
        rows.append({"cell_type":ct,"axis":ax,"status":"축 없음"}); continue
    ci,ki=types.index(ct),PROGRAMS.index(ax)
    coll=float(np.corrcoef(st_mean[:,ci,ki].numpy(),pit)[0,1])
    w=float(W[ci,ki])
    ok=(np.sign(w)==sign)
    excluded=abs(coll)>0.3
    rows.append({"cell_type":ct,"axis":ax,"expected":"+" if sign>0 else "−","fitted_W":round(w,4),
                 "agrees":bool(ok),"corr_with_pi_tumor":round(coll,3),
                 "excluded_collinear":bool(excluded),"rationale":why})
    if not excluded: keep.append((ci,ki,sign))

obs=float(np.mean([np.sign(float(W[c,k]))==s for c,k,s in keep])) if keep else float("nan")
rng=np.random.default_rng(11); null=[]
for i in range(R_NULL):
    o=rng.permutation(len(tr))
    Wp=fitW(time[torch.tensor(o,device=dev)],event[torch.tensor(o,device=dev)])
    null.append(float(np.mean([np.sign(float(Wp[c,k]))==s for c,k,s in keep])))
    if (i+1)%50==0: print(f"[{ORG}] null {i+1}/{R_NULL}",flush=True)
null=np.asarray(null)
out={"organ":ORG,"model":"E2''","n_expectations":len(EXPECT[ORG]),"n_tested":len(keep),
     "n_excluded_collinear":sum(1 for r in rows if r.get("excluded_collinear")),
     "concordance":round(obs,4),"null_mean":round(float(null.mean()),4),
     "null_sd":round(float(null.std()),4),
     "p":round(float((np.sum(null>=obs)+1)/(R_NULL+1)),4),
     "R_null":R_NULL,"per_expectation":rows}
print(json.dumps({k:v for k,v in out.items() if k!="per_expectation"},indent=1),flush=True)
TAG=os.environ.get("BIO_TAG","")
open(f"results/report/BIO_CONCORDANCE_{ORG}{TAG}.json","w").write(json.dumps(out,ensure_ascii=False,indent=1))
