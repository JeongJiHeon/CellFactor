#!/usr/bin/env python3
"""Figure 2 KM 원자료 — km_acand3k.py(→ km_v6.py)와 같은 위험·종점 정의로 5모델 × 4코호트의 (위험, 시간, 사건) 배열을 저장(그림·검정 없음).
출력 final_results/04_prognosis/KM_5MODELS_ARRAYS.npz (키 = '{코호트}|{모델}|{risk,time,event}')"""
import os, re, numpy as np
src_a = open("analysis/report/acand3k/km_acand3k.py").read()
pre = src_a[:src_a.index('src = open("analysis/report/km_v6.py").read()')]
g = {"__name__": "dump"}; exec(compile(pre, "km_acand3k(pre)", "exec"), g)
bp = g["build"]("breast", "BREAST_ACAND3K_RISKS.json"); lp = g["build"]("liver", "LIVER_ACAND3K_RISKS.json"); pp = g["build"]("pancreas", "PANC_ACAND3K_RISKS.json")
os.environ["KM_BR_CP"] = bp
src = open("analysis/report/km_v6.py").read().replace('J("results/report/LIVER_V5_RISKS.json")', f'J("{lp}")').replace('J("results/report/PANC_V5_RISKS.json")', f'J("{pp}")')
src = src[:src.index("fig,axes=plt.subplots")]
h = {"__name__": "dump"}; exec(compile(src, "km_v6(arrays)", "exec"), h)
out = {}
for cs, fns in h["CASES"]:
    for m in h["MODELS"]:
        got = fns[m]()
        if got is None: continue
        r, t, e = got; name = "CellFactor" if m == "CP-PCH" else m
        out[f"{cs}|{name}|risk"], out[f"{cs}|{name}|time"], out[f"{cs}|{name}|event"] = r, t, e
np.savez_compressed("final_results/04_prognosis/KM_5MODELS_ARRAYS.npz", **out); print(len(out) // 3, "arrays")
