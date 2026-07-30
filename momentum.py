import numpy as np
from linklet_analysis.config import REFRACTIVE_CORRECTION,RADIATION_LENGTH
electron_mass = 0.511  # MeV

# 引数として受け取った DataFrame に運動量とその誤差などの計算結果を付け足して返す関数
def calculate_momentum_old(df,down_layer,n_layers):
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
    dax = np.atan(ax1*np.cos(track_B0s)*REFRACTIVE_CORRECTION) - np.atan(ax0*np.cos(track_B0s)*REFRACTIVE_CORRECTION) # 屈折率による浮き上がりの作用を打ち消す
    day = np.atan(ay1*np.cos(track_A0s)*REFRACTIVE_CORRECTION) - np.atan(ay0*np.cos(track_A0s)*REFRACTIVE_CORRECTION)

    for idx, layer_num in enumerate(linklet_range):
        df[f"dax_{layer_num}"] = dax[:, idx]
        df[f"day_{layer_num}"] = day[:, idx]

    # if cut==1:
    #     dax[:,3]=dax[:,3]-0.01
    #     day[:,3]=day[:,3]-0.01

    thick = 0.007*2 / RADIATION_LENGTH #放射長3.35 cm 乳剤層の厚み 70 μm
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


def calculate_momentum(df,start_layer,n_layers,scan_direction=True):
    df = df.copy()

    n_linklets = n_layers - 1
    if scan_direction:
        end_layer = start_layer + n_layers
    else:
        end_layer = start_layer - n_layers
    
    linklet_range = range(min(start_layer,end_layer) + 1, max(start_layer,end_layer))

    ax0 = df[[f"ax0_{i}{i+1}" for i in linklet_range]].to_numpy()
    ax1 = df[[f"ax1_{i}{i+1}" for i in linklet_range]].to_numpy()
    ay0 = df[[f"ay0_{i}{i+1}" for i in linklet_range]].to_numpy()
    ay1 = df[[f"ay1_{i}{i+1}" for i in linklet_range]].to_numpy()
    print(f"ax shape is :{ax0.shape} [events,linklets]")
    if ax0.shape[1] != n_linklets:
        raise ValueError(f"リンクレット数が不一致です: expected={n_linklets}, actual={ax0.shape[1]}")

    
    # 入射角 (A, B) の計算
    # 最上流の層の傾き（up_layer=up_linkletのax1）からビーム入射角を算出
    if scan_direction:
        track_ax0_slope = ax0[:, 0]
        track_ay0_slope = ay0[:, 0]
    else:
        track_ax0_slope = ax1[:, -1]
        track_ay0_slope = ay1[:, -1]

    track_ax0_theta = np.arctan(track_ax0_slope)
    track_ay0_theta = np.arctan(track_ay0_slope)

    # DataFrameには1次元配列を保存
    df["track_up_ax_slope"] = track_ax0_slope
    df["track_up_ay_slope"] = track_ay0_slope

    # 3次元幾何補正を考慮した角度差（散乱角）の計算 [rad]
    theta_x_2d = track_ax0_theta[:, None]
    theta_y_2d = track_ay0_theta[:, None]
    dax = np.atan(ax1*np.cos(theta_y_2d)*REFRACTIVE_CORRECTION) - np.atan(ax0*np.cos(theta_y_2d)*REFRACTIVE_CORRECTION) # 屈折率による浮き上がりの作用を打ち消す
    day = np.atan(ay1*np.cos(theta_x_2d)*REFRACTIVE_CORRECTION) - np.atan(ay0*np.cos(theta_x_2d)*REFRACTIVE_CORRECTION)

    for idx, layer_num in enumerate(linklet_range):
        df[f"dax_{layer_num}{layer_num+1}"] = dax[:, idx]
        df[f"day_{layer_num}{layer_num+1}"] = day[:, idx]
        
    N1 = dax.shape[1]
    N2 = dax.shape[1] + day.shape[1]

    # if cut==1:
    #     dax[:,3]=dax[:,3]-0.01
    #     day[:,3]=day[:,3]-0.01

    thick = 0.007*2 / RADIATION_LENGTH #放射長3.35 cm 乳剤層の厚み 70 μm
    norm = np.sqrt(1.0 + track_ax0_slope**2 + track_ay0_slope**2)
    cos_theta_z = 1.0 / norm
    effective_thick = thick / cos_theta_z  # 入射角を考慮した飛跡ごとの有効厚み
    
    # 多重散乱の式の対数項(z=1,β=1)
    log_term = 1.0 + 0.038 * np.log(effective_thick)
    sqrt_thick = np.sqrt(effective_thick)

    #--- 角度ずれRMSの計算 ---
    df["RMS"] = np.sqrt((np.mean(dax**2, axis=1) + np.mean(day**2, axis=1)) / 2) # XとYの両方の散乱角の二乗和から全体のRMSを計算, axis=1（層方向）で平均をとる
    df["RMS_e"] = df["RMS"] / np.sqrt(2 * N2)  # 独立な正規分布を仮定した統計誤差
    #--- 運動量の計算 ---  電磁散乱の式を逆算
    df["p"] = (13.6 / df["RMS"]) * sqrt_thick * log_term # xyすべての角度ずれのRMSから算出
    df["p_e"] = df["p"] / np.sqrt(2 * N2)      # 独立な正規分布を仮定した統計誤差

    # ---------------------------------------------------------
    # トラックの3次元運動量ベクトル
    # ---------------------------------------------------------
    # trackの3次元単位ベクトルの計算
    ux = track_ax0_slope / norm
    uy = track_ay0_slope / norm
    uz = 1.0 / norm

    df["ux"] = ux
    df["uy"] = uy
    df["uz"] = uz

    # 運動量各ベクトル成分
    df["Px"] = df["p"] * ux
    df["Py"] = df["p"] * uy
    df["Pz"] = df["p"] * uz

    # 電子or陽電子だと思った時のエネルギー
    df["E"] = np.sqrt(df["p"]**2 + electron_mass**2)
    df["E_e"] = df["p"] / df["E"] * df["p_e"]

    # ---------------------------------------------------------
    # 診断用情報
    # ---------------------------------------------------------
    df["RMSx"] = np.sqrt(np.mean(dax**2, axis=1)) # RMS, RMSx, RMSy は同じ分布になるはず
    df["RMSy"] = np.sqrt(np.mean(day**2, axis=1))
    df["RMSx_e"] = df["RMSx"] / np.sqrt(2 * N1)
    df["RMSy_e"] = df["RMSy"] / np.sqrt(2 * N1)

    df["p_from_x"] = (13.6 / df["RMSx"]) * sqrt_thick * log_term  # p, p_from_x, p_from_y は同じ分布になるはず
    df["p_from_y"] = (13.6 / df["RMSy"]) * sqrt_thick * log_term

    df["p_from_x_e"] = df["p_from_x"] / np.sqrt(2 * N1)
    df["p_from_y_e"] = df["p_from_y"] / np.sqrt(2 * N1)

    n_total = len(df)
    n_success = np.isfinite(df["p"]).sum()
    print(f"Calculated: {n_success}/{n_total} events")
    return df
