N_LAYERS = 5
CELL_SIZE = 5.0
REFRACTIVE_CORRECTION = 0.95 #屈折率
RADIATION_LENGTH = 3.35 #乳剤層の放射長

# ======================
# カットまとめ
# ======================
CUT_REGISTRY = {
    0: {"name": "non",
        "rule":{}},
    1: {"name": "standard", 
        "rule":{"ax":0.0, "ax_sigma":1.0, "ay":0.0, "ay_sigma":1.0, "ph_th":40}}
}
