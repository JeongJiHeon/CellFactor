"""공간 그림 공통 병리 라벨 색 — 고위험 라벨 = 주황 계열, 저위험 라벨 = 초록 계열, 그 외 라벨 = 중립(베이지·회색·연보라) 계열.
고위험/저위험 묶음은 단일세포 고위험 구성(final_results/01_umap)에서 정한 것(fig_spatial_groups.py 와 같음)."""
from matplotlib.patches import Patch

UNLABELLED = "#f6f5f2"
COLOR = {
    # 고위험 라벨
    "Tumor": "#eb6834", "in situ": "#a8431a", "Vessels": "#f4a37c",
    "invasive cancer": "#eb6834", "cancer in situ": "#a8431a",
    # 저위험 라벨
    "Lymphocyte": "#1baf7a", "High TIL stroma": "#0f7a52", "Lymphoid nodule": "#7fd6b1", "Lactiferous duct": "#3f9e78",
    "Adj. liver": "#1baf7a", "Acinar/islet": "#1baf7a", "immune infiltrate": "#1baf7a", "breast glands": "#3f9e78",
    # 그 외(기대 방향 없음)
    "Acellular stroma": "#e4ddcf", "Stroma cell": "#c8b796", "Low TIL stroma": "#b8c2cb", "Necrosis": "#6f6c66",
    "Fat tissue": "#f2e7b6", "Tumor region": "#d8c4b6", "Heterologous elements": "#a9a2bb", "Nerve": "#8f89a5",
    "Stroma": "#c8b796", "connective tissue": "#c8b796", "adipose tissue": "#f2e7b6", "Other": "#cfcec9",
}
NAME = {"Tumor": "Tumour", "in situ": "Tumour in situ", "Vessels": "Vessels", "High TIL stroma": "High-TIL stroma",
        "Low TIL stroma": "Low-TIL stroma", "Lactiferous duct": "Lactiferous duct (normal)", "Adj. liver": "Adjacent liver",
        "invasive cancer": "Invasive cancer", "cancer in situ": "Cancer in situ", "immune infiltrate": "Immune infiltrate",
        "breast glands": "Breast glands (normal)", "connective tissue": "Connective tissue", "adipose tissue": "Adipose tissue",
        "Tumor region": "Tumour region"}


def color(label, labelled=True):
    return COLOR.get(label, "#cfcec9") if labelled else UNLABELLED


def handles(counts, high, low, fontsize_hint=None):
    """counts: {label: n spots in this section}. 고위험 → 저위험 → 그 외(많은 순) 순서, 'n' 표기."""
    hi = [L for L in high if counts.get(L, 0) > 0]
    lo = [L for L in low if counts.get(L, 0) > 0]
    ot = sorted([L for L in counts if L not in high and L not in low and counts[L] > 0], key=lambda L: -counts[L])
    return [Patch(color=COLOR.get(L, "#cfcec9"), label=f"{NAME.get(L, L)} ({counts[L]})") for L in hi + lo + ot]
