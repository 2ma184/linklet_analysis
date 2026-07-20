import pandas as pd
import numpy as np

import numpy as np

def calc_vertex(P_A, V_A, P_B, V_B):
    """
    2本の3次元直線
        L_A = P_A + s V_A
        L_B = P_B + t V_B
    の最近接点とその中点(Vertex)を返す
    """

    w0 = P_A - P_B

    a = np.dot(V_A, V_A)
    b = np.dot(V_A, V_B)
    c = np.dot(V_B, V_B)
    d = np.dot(V_A, w0)
    e = np.dot(V_B, w0)

    det_nom = a*c - b*b

    if abs(det_nom) < 1e-12: #det_nomが0なら平行
        return np.nan, np.array([np.nan, np.nan, np.nan])
    
    #二つの直線の最近接点でのs,t
    s = (b*e - c*d) / det_nom 
    t = (a*e - b*d) / det_nom

    C_A = P_A + s*V_A
    C_B = P_B + t*V_B

    vertex = 0.5*(C_A + C_B)
    closest_distance = np.linalg.norm(C_A - C_B)

    return closest_distance, vertex



def extract_gamma_events(df, start_layer, n_layers, small_is_up=True):
    #例は全て start_layer=5, n_layer=5, small_is_up=Faleseの時
    end_layer = start_layer - n_layers + 1 # 例: 5 - 5 + 1 = 1

    # ---------------------------------------------------------
    # データフレームを辞書リストに変換
    # ---------------------------------------------------------
    tracks = df.to_dict(orient="records")
    
    CELL = 5.0  # 近接探索用のセルの大きさ (5mm)??5um??
    grid = {}
    
    # 一番上流端のリンク座標（例: x1_45）を基準に空間ハッシュを作成
    for t in tracks:
        ix = int(np.floor(t[f"x1_{start_layer-1}{start_layer}"] / CELL))
        iy = int(np.floor(t[f"y1_{start_layer-1}{start_layer}"] / CELL))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(t)

    seen_pairs = set()
    gamma_events = []
    pair_count = 0

    print(f"Starting grid-based pair collision check (Link {start_layer-1}{start_layer} as Downstream Base)...")
 
    # 各セルをループ
    for cell_key, vecA in grid.items():
        ixA, iyA = cell_key
        
        for A in vecA:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (ixA + dx, iyA + dy)
                    if neighbor_key not in grid:
                        continue
                    
                    for B in grid[neighbor_key]:
                        pair_count += 1
                        print(f"\rChecked pairs: {pair_count}", end="", flush=True)

                        # =========================================================
                        # ① IDベースの重複排除
                        # ========================================================
                        a_id = int(A[f"id{start_layer}"])
                        b_id = int(B[f"id{start_layer}"])
                        
                        if A is B: #A,Bが同じなら飛ばす
                            continue
                            
                        pair_key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                        if pair_key in seen_pairs:
                            continue #見たことあるidなら飛ばす

                        # =========================================================
                        # ② γ線の物理条件カット（幾何学的収束・ねじれ判定）
                        # =========================================================
                        # リンク番号の開始から終了までループを回して距離を取得
                        #distance track center??
                        dtc_0 = []
                        dtc_1 = []
                        for idx in range(end_layer, start_layer): # 例: start_layer=5, n_layers=5 のとき、idx は 1, 2, 3, 4 と動く
                            dtc_0.append(np.hypot(A[f"x0_{idx}{idx+1}"] - B[f"x0_{idx}{idx+1}"], A[f"y0_{idx}{idx+1}"] - B[f"y0_{idx}{idx+1}"])) 
                            dtc_1.append(np.hypot(A[f"x1_{idx}{idx+1}"] - B[f"x1_{idx}{idx+1}"], A[f"y1_{idx}{idx+1}"] - B[f"y1_{idx}{idx+1}"])) # 配列の末尾 ([-1]) が 最上流 (idx=4,x1_45) のデータになる

                        # カット1: 上流端（対生成の根本）のトラック同士の距離が5um以下
                        if dtc_1[-1] > 5.0: 
                            continue

                        # カット2: 下流（配列の末尾）に向かって2本の距離が狭まっていること
                        # 上流から下流へ、値が「単調増増加」しているかチェック (dtc[idx] > dt_next)
                        if not all(dtc_0[idx] > dt0_next for idx, dt0_next in enumerate(dtc_0[1:])):
                            continue
                        if not all(dtc_1[idx] > dt1_next for idx, dt1_next in enumerate(dtc_1[1:-1])):
                            continue
                        if not (dtc_1[-1] < dtc_0[-1] < dtc_0[-2]): # 根本付近の厳密な順序チェック,dct_1_45 < dtc_0_45 < dtc_0_34
                            continue

                        # カット3: 上流側（配列の先頭側）では十分離れていること
                        if not (dtc_1[-2] > 20 and dtc_1[-3] > 30 and dtc_1[-4] > 30):
                            continue

                        # カット4: ねじれ（交差）チェック
                        # 下流端の中点(x0_12,y0_12)を基準にする
                        mx = (A[f"x0_{end_layer}{end_layer+1}"] + B[f"x0_{end_layer}{end_layer+1}"]) / 2.0
                        my = (A[f"y0_{end_layer}{end_layer+1}"] + B[f"y0_{end_layer}{end_layer+1}"]) / 2.0

                        # 各リンクでのねじれ条件
                        twisted = False
                        for idx in range(end_layer+1, start_layer): # 例: start_layer=5, n_layers=5 のとき、idx は 2, 3, 4 と動く
                            val_0 = np.hypot(A[f"x0_{idx}{idx+1}"] - mx, A[f"y0_{idx}{idx+1}"] - my) - np.hypot(B[f"x0_{idx}{idx+1}"] - mx, B[f"y0_{idx}{idx+1}"] - my)
                            val_1 = np.hypot(A[f"x1_{idx}{idx+1}"] - mx, A[f"y1_{idx}{idx+1}"] - my) - np.hypot(B[f"x1_{idx}{idx+1}"] - mx, B[f"y1_{idx}{idx+1}"] - my)
                            if val_1 * val_0 < 0: #val_0とval1の符号が逆ならねじれ
                                twisted = True
                                break #一つでもねじれていたらcut
                        if twisted:
                            continue

                        seen_pairs.add(pair_key)

                        # =========================================================
                        # ③ γ線物理量の算出
                        # =========================================================
                        trackA_p = A["p"]
                        trackB_p = B["p"]
                        trackA_p_e = A["p_e"]
                        trackB_p_e = B["p_e"]

                        trackA_ax0, trackA_ay0 = A["track_up_ax"], A["track_up_ay"]
                        trackB_ax0, trackB_ay0 = B["track_up_ax"], B["track_up_ay"]

                        # 運動量・角度の合成
                        gamma_p = trackA_p + trackB_p
                        gamma_p_e = np.sqrt((trackA_p_e ** 2) + (trackB_p_e ** 2)) 

                        gamma_Px = A["px"] + B["px"]
                        gamma_Py = A["py"] + B["py"]
                        gamma_P_xy = np.sqrt((gamma_Px ** 2) + (gamma_Py ** 2)) # ここ２乗平均ではなく平均では？

                        gamma_Px_e = np.sqrt((A["px_e"] ** 2) + (B["px_e"] ** 2))
                        gamma_Py_e = np.sqrt((A["py_e"] ** 2) + (B["py_e"] ** 2))
                        gamma_P_xy_e = np.sqrt(((gamma_Px_e**2 * gamma_Px**2) + (gamma_Py_e**2 * gamma_Py**2)) / (gamma_Px**2 + gamma_Py**2)) #ここも改変必要？

                        gamma_ax = (trackA_p * trackA_ax0 + trackB_p * trackB_ax0) / gamma_p #運動量に重みをつけた角度
                        gamma_ay = (trackA_p * trackA_ay0 + trackB_p * trackB_ay0) / gamma_p

                        # 誤差の伝播計算
                        dXdP_A = (trackA_ax0 - trackB_ax0) * trackB_p / (gamma_p ** 2)
                        dXdP_B = (trackB_ax0 - trackA_ax0) * trackA_p / (gamma_p ** 2)
                        dYdP_A = (trackA_ay0 - trackB_ay0) * trackB_p / (gamma_p ** 2)
                        dYdP_B = (trackB_ay0 - trackA_ay0) * trackA_p / (gamma_p ** 2)
                        
                        gamma_ax_e = np.sqrt((dXdP_A**2 * trackA_p_e**2) + (dXdP_B**2 * trackB_p_e**2))
                        gamma_ay_e = np.sqrt((dYdP_A**2 * trackA_p_e**2) + (dYdP_B**2 * trackB_p_e**2))

                        # 開き角 (Opening Angle)
                        open_angle_x = np.abs(trackA_ax0 - trackB_ax0)
                        open_angle_y = np.abs(trackA_ay0 - trackB_ay0)
                        open_angle = np.sqrt((trackA_ax0 - trackB_ax0)**2 + (trackA_ay0 - trackB_ay0)**2)

                        #gamma線のpair同士の距離
                        track_distance1 = dtc_1[-1]
                        track_distance2 = dtc_1[-2]
                        track_distance3 = dtc_1[-3]
                        track_distance4 = dtc_1[-4]

                        # =====================================================
                        # Vertex (最近接点)
                        # =====================================================
                        start_layer_pointA = np.array([A[f"x1_{start_layer-1}{start_layer}"], A[f"y1_{start_layer-1}{start_layer}"], A[f"z1_{start_layer-1}{start_layer}"] ], dtype=float)
                        start_layer_pointB = np.array([B[f"x1_{start_layer-1}{start_layer}"], B[f"y1_{start_layer-1}{start_layer}"], B[f"z1_{start_layer-1}{start_layer}"] ], dtype=float)

                        start_layer_vectorA = np.array([trackA_ax0, trackA_ay0, 1.0 ], dtype=float)
                        start_layer_vectorB = np.array([trackB_ax0, trackB_ay0, 1.0 ], dtype=float)

                        track_distance0, vertex = calc_vertex(start_layer_pointA, start_layer_vectorA, start_layer_pointB, start_layer_vectorB)

                        vertex_x = vertex[0]
                        vertex_y = vertex[1]
                        vertex_z = vertex[2]

                        # =========================================================
                        # ④ イベント辞書の構築
                        # =========================================================
                        event = {}
                        for k, v in A.items():
                            event[f"{k}_a"] = v
                        for k, v in B.items():
                            event[f"{k}_b"] = v

                        # 特有の物理計算結果を上書き・追加
                        event.update({
                            "RMSx_a": A["RMSx"], "RMSy_a": A["RMSy"],
                            "RMSx_b": B["RMSx"], "RMSy_b": B["RMSy"],
                            "RMS_a": A["RMS"], "RMS_a_e": A["RMS_e"],
                            "RMS_b": B["RMS"], "RMS_b_e": B["RMS_e"],

                            "Px_a": A["px"], "Py_a": A["py"],
                            "Px_b": B["px"], "Py_b": B["py"],
                            "trackA_p": trackA_p, "trackB_p": trackB_p, "gamma_p": gamma_p,
                            "P_xy_a": A["p_xy"], "P_xy_b": B["p_xy"], "gamma_P_xy": gamma_P_xy,

                            "Px_a_e": A["px_e"], "Py_a_e": A["py_e"],
                            "Px_b_e": B["px_e"], "Py_b_e": B["py_e"],
                            "trackA_p_e": trackA_p_e, "trackB_p_e": trackB_p_e, "gamma_p_e": gamma_p_e,
                            "P_xy_a_e": A["p_xy_e"], "P_xy_b_e": B["p_xy_e"], "gamma_P_xy_e": gamma_P_xy_e,

                            "open_angle": open_angle, "open_angle_x": open_angle_x, "open_angle_y": open_angle_y,

                            "gamma_ax": gamma_ax, "gamma_ay": gamma_ay,
                            "gamma_ax_e": gamma_ax_e, "gamma_ay_e": gamma_ay_e,

                            "trackA_ax0": trackA_ax0, "trackA_ay0": trackA_ay0, "trackB_ax0": trackB_ax0, "trackB_ay0": trackB_ay0,

                            "track_distance1": track_distance1, "track_distance2": track_distance2, "track_distance3": track_distance3, "track_distance4": track_distance4,

                            "track_distance0": track_distance0, "vertex_x": vertex_x, "vertex_y": vertex_y, "vertex_z": vertex_z
                        })
                        
                        gamma_events.append(event)

    print("\n=== 終了 ===")
    n_events = len(gamma_events)
    n_events_err = np.sqrt(n_events)
    print(f"抽出されたγ線イベント数: {n_events} ± {n_events_err:.1f}")

    return pd.DataFrame(gamma_events)


