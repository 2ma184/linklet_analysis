CELL_SIZE = 5.0
REFRACTIVE_CORRECTION = 0.95 #屈折率
RADIATION_LENGTH = 3.35 #乳剤層の放射長

# ======================
# QST2021カットまとめ
# ======================
CUT_REGISTRY_QST2021 = {
    0: {"name": "non",
        "rule":{}},
    1: {"name": "standard", 
        "rule":{"ax":0.0, "ax_sigma":1.0, "ay":0.0, "ay_sigma":1.0, "ph_th":40}}
}


# ======================
# KAKENカットまとめ
# ======================
CUT_REGISTRY_KAKEN1 = {
    0: {"name": "50MeV 0deg",    "Energy" : 50, "deg" : 0,
        "rule":{"cx":22, "cy":20, "r":20, "ax":-0.03, "ax_sigma":0.15, "ay":-0.00, "ay_sigma":0.1, "ph_th":50}},
    
    1: {"name": "50MeV 22.5deg", "Energy" : 50, "deg" : 22.5,
        "rule":{"cx":20, "cy":63, "r":20, "ax":-0.03, "ax_sigma":0.15, "ay":-0.42, "ay_sigma":0.1, "ph_th":45}},
        
    2: {"name": "50MeV 45deg",   "Energy" : 50, "deg" : 45,
        "rule":{"cx":20,"cy":100, "r":30, "ax":-0.03, "ax_sigma":0.15, "ay":-1.00, "ay_sigma":0.1, "ph_th":40}},

    3: {"name": "40MeV 0deg",    "Energy" : 40, "deg" : 0,
        "rule":{"cx":65, "cy":62, "r":20, "ax":-0.03, "ax_sigma":0.15, "ay":-0.00, "ay_sigma":0.1, "ph_th":50}},
    
    4: {"name": "40MeV 22.5deg", "Energy" : 40, "deg" : 22.5,
        "rule":{"cx":65,"cy":100, "r":20, "ax":-0.03, "ax_sigma":0.15, "ay":-0.42, "ay_sigma":0.1, "ph_th":45}},
        
    5: {"name": "40MeV 45deg",   "Energy" : 40, "deg" : 45,
        "rule":{"cx":65, "cy":22, "r":30, "ax":-0.03, "ax_sigma":0.15, "ay":-1.00, "ay_sigma":0.1, "ph_th":40}},
}

CUT_REGISTRY_KAKEN2 = {
    0: {"name": "30MeV 0deg",    "Energy" : 30, "deg" : 0,
        "rule":{"cx":22, "cy":16, "r":20, "ax":-0.03, "ax_sigma":0.1, "ay":-0.0, "ay_sigma":0.1,  "ph_th":50}},
    
    1: {"name": "30MeV 22.5deg", "Energy" : 30, "deg" : 22.5,
        "rule":{"cx":25, "cy":60, "r":20, "ax":-0.03, "ax_sigma":0.1, "ay":-0.42, "ay_sigma":0.1,  "ph_th":45}},
        
    2: {"name": "30MeV 45deg",   "Energy" : 30, "deg" : 45,
        "rule":{"cx":27,"cy":100, "r":30, "ax":-0.03, "ax_sigma":0.1, "ay":-1.0,  "ay_sigma":0.15, "ph_th":40}},

    3: {"name": "20MeV 0deg",    "Energy" : 20, "deg" : 0,
        "rule":{"cx":65, "cy":55, "r":20, "ax":-0.03, "ax_sigma":0.1, "ay":-0.0,  "ay_sigma":0.1,  "ph_th":50}},
    
    4: {"name": "20MeV 22.5deg", "Energy" : 20, "deg" : 22.5,
        "rule":{"cx":70, "cy":100, "r":20, "ax":-0.03, "ax_sigma":0.1, "ay":-0.42, "ay_sigma":0.1,  "ph_th":45}}, 
        
    5: {"name": "20MeV 45deg",   "Energy" : 20, "deg" : 45,
        "rule":{"cx":67, "cy":30, "r":30, "ax":-0.03, "ax_sigma":0.1, "ay":-1.0,  "ay_sigma":0.15, "ph_th":40}},
}

