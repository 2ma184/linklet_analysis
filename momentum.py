import numpy as np

# 引数として受け取った DataFrame に運動量とその誤差などの計算結果を付け足して返す関数
def calculate_momentum(df,down_layer,n_layers):
    up_layer = down_layer + n_layers -1

    down_linklet = down_layer
    n_linklets = n_layers - 1
    up_linklet = down_linklet + n_linklets -1
    linklet_range = range(down_linklet, up_linklet + 1)

    N1 = n_linklets
    N2 = 2*n_linklets

    ax0 = df[[f"ax0_{i}" for i in linklet_range]].to_numpy()
    ax1 = df[[f"ax1_{i}" for i in linklet_range]].to_numpy()
    ay0 = df[[f"ay0_{i}" for i in linklet_range]].to_numpy()
    ay1 = df[[f"ay1_{i}" for i in linklet_range]].to_numpy()
    print(f"ax shape is :{ax0.shape}")

    # 入射角 (A, B) の計算
    # 最上流の層の傾き（up_layer=up_linkletのax1）からビーム入射角を算出
    track_A0 = np.atan(ax1[:,-1]) #傾きからradに直す
    track_A0s = track_A0[:, None]
    track_B0 = np.atan(ay1[:,-1])
    track_B0s = track_B0[:, None]
    df["track_up_ax"] = track_A0s
    df["track_up_ay"] = track_B0s

    # 3次元幾何補正を考慮した角度差（散乱角）の計算
    dax = np.atan(ax1*np.cos(track_B0s)*0.95) - np.atan(ax0*np.cos(track_B0s)*0.95) # 0.95は屈折率による浮き上がりの作用を打ち消す
    day = np.atan(ay1*np.cos(track_A0s)*0.95) - np.atan(ay0*np.cos(track_A0s)*0.95)

    thick = 0.007*2 / 3.35 #放射長3.35 cm 乳剤層の厚み 70 μm
    cos_track0 = np.cos(np.atan(np.sqrt(np.tan(track_A0)**2+np.tan(track_B0)**2)))
    effective_thick = thick / cos_track0 # 1飛跡ごとの有効厚み
    
    # 多重散乱の式の対数項
    log_term = 1.0 + 0.038 * np.log(effective_thick)
    sqrt_thick = np.sqrt(effective_thick)

    #--- 角度ずれRMSの計算 ---
    df["RMSx"] = np.sqrt(np.mean(dax**2, axis=1)) # axis=1（層方向）で平均をとる
    df["RMSy"] = np.sqrt(np.mean(day**2, axis=1))
    df["RMS"] = np.sqrt((np.mean(dax**2, axis=1) + np.mean(day**2, axis=1)) / 2) # XとYの両方の散乱角の二乗和から全体のRMSを計算
    # 統計誤差の計算
    df["RMSx_e"] = df["RMSx"] / np.sqrt(2 * (N1 - 1))
    df["RMSy_e"] = df["RMSy"] / np.sqrt(2 * (N1 - 1))
    df["RMS_e"] = df["RMS"] / np.sqrt(2 * (N2 - 1))


    #--- 運動量の計算 ---
    df["px"] = (13.6 / df["RMSx"]) * sqrt_thick * log_term  # 電磁散乱の式を逆算
    df["py"] = (13.6 / df["RMSy"]) * sqrt_thick * log_term
    df["p_xy"] = (df["px"] + df["py"]) / 2 # ただの平均
    df["p"] = (13.6 / df["RMS"]) * sqrt_thick * log_term # xyのRMSから算出
    # 統計誤差の計算
    df["px_e"] = df["px"] / np.sqrt(2 * (N1 - 1))
    df["py_e"] = df["py"] / np.sqrt(2 * (N1 - 1))
    df["p_xy_e"] = 0.5 * np.sqrt(df["px_e"]**2 + df["py_e"]**2)
    df["p_e"] = df["p"] / np.sqrt(2 * (N2 - 1))

    print(f"calculated : {len(df)}")
    return df
