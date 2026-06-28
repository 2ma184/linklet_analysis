import numpy as np

# ======================
# カットまとめ
# ======================
CUT_REGISTRY = {
    0: {"name": "non",
        "rule":{}},
    1: {"name": "standard", 
        "rule":{"ax":0.0, "ax_sigma":1.0, "ay":0.0, "ay_sigma":1.0, "ph_th":40}}
}

# =========================================================
# カット関数群
# =========================================================
def cut_xy(cx, cy, r, x, y):
    return ((x/1000.0 - cx)**2 + (y/1000.0 - cy)**2) < r**2

def cut_ph(ph0, ph1, threshold=50):
    return (ph0 + ph1)/10000.0 > threshold

def cut_theta(center, threshold, theta):
    return ((theta - center)**2 < threshold**2)

#-- CUT適応関数
def apply_cut(df, cut_rule): 
    # 最初はすべて True（全行通過）のマスクを作成    
    mask = np.ones(len(df), dtype=bool)
    # ビーム位置
    if "cx" in cut_rule and "cy" in cut_rule and "r" in cut_rule: 
        mask &= cut_xy(cut_rule["cx"], cut_rule["cy"], cut_rule["r"], df["x0"], df["y0"])
    # ビーム角度x
    if "ax" in cut_rule and "ax_sigma" in cut_rule: 
        mask &= cut_theta(cut_rule["ax"], cut_rule["ax_sigma"], df["ax0"])
        mask &= cut_theta(cut_rule["ax"], cut_rule["ax_sigma"], df["ax1"])     
    # ビーム角度y
    if "ay" in cut_rule and "ay_sigma" in cut_rule:
        mask &= cut_theta(cut_rule["ay"], cut_rule["ay_sigma"], df["ay0"])
        mask &= cut_theta(cut_rule["ay"], cut_rule["ay_sigma"], df["ay1"])
    # パルスハイト
    if "ph_th" in cut_rule: 
        mask &= cut_ph(df["ph0"], df["ph1"], cut_rule["ph_th"])
    return df[mask]

