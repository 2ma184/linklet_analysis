import pandas as pd
import numpy as np

def extract_gamma_events(input_csv_path, output_csv_path, start_layer, n_layers):
    print(f"Loading momentum-calculated CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Total available tracks: {len(df)}")

    # プレート数が5のとき、リンク（カラム名）の終端は4になる
    end_layer = start_layer + n_layers - 1  # 例: 1 + 5 - 1 = 5
    end_link = end_layer - 1                # 例: 5 - 1 = 4 
    
    # ---------------------------------------------------------
    # データフレームを辞書リストに変換
    # ---------------------------------------------------------
    tracks = df.to_dict(orient="records")
    
    CELL = 5.0  # 近接探索用のセルの大きさ (5mm)
    grid = {}
    
    # 一番上流端のリンク座標（例: x1_4）を基準に空間ハッシュを作成
    for t in tracks:
        ix = int(np.floor(t[f"x1_{end_link}"] / CELL))
        iy = int(np.floor(t[f"y1_{end_link}"] / CELL))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(t)

    seen_pairs = set()
    gamma_events = []
    pair_count = 0

    print(f"Starting grid-based pair collision check (Link {end_link} as Downstream Base)...")

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
                        a_id = int(A[f"id{end_layer}"])
                        b_id = int(B[f"id{end_layer}"])
                        
                        if A is B:
                            continue
                            
                        p_key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                        if p_key in seen_pairs:
                            continue

                        # =========================================================
                        # ② 物理条件カット（幾何学的収束・ねじれ判定）
                        # =========================================================
                        # リンク番号の開始から終了までループを回して距離を取得
                        # 例: start_layer=1, n_layers=5 のとき、idx は 1, 2, 3, 4 と動く
                        dtc0 = []
                        dtc1 = []
                        for idx in range(start_layer, end_layer):
                            dtc0.append(np.hypot(A[f"x0_{idx}"] - B[f"x0_{idx}"], A[f"y0_{idx}"] - B[f"y0_{idx}"]))
                            dtc1.append(np.hypot(A[f"x1_{idx}"] - B[f"x1_{idx}"], A[f"y1_{idx}"] - B[f"y1_{idx}"]))

                        # カット1: 終端リンク（下流端・対生成の根本）の出口距離が5mm以下
                        # 配列の末尾 [-1] が end_link (4番) のデータになります
                        if dtc1[-1] > 5.0: 
                            continue

                        # カット2: 下流（配列の末尾）に向かって2本の距離が狭まっていること（収束条件）
                        # 上流から下流へ、値が「単調減少」しているかチェック (dtc[idx] > dt_next)
                        if not all(dtc0[idx] > dt0_next for idx, dt0_next in enumerate(dtc0[1:])):
                            continue
                        if not (dtc1[-1] < dtc0[-1] < dtc0[-2]): # 根本付近の厳密な順序チェック
                            continue
                        if not all(dtc1[idx] > dt1_next for idx, dt1_next in enumerate(dtc1[1:-1])):
                            continue

                        # カット3: 上流側（配列の先頭側）では十分離れていること
                        # 4層（リンク4つ分）ある場合の、上流側の閾値判定
                        if len(dtc1) >= 4:
                            if not (dtc1[-2] > 20 and dtc1[-3] > 30 and dtc1[-4] > 30):
                                continue
                        elif len(dtc1) == 3:
                            if not (dtc1[-2] > 20 and dtc1[-3] > 30):
                                continue

                        # カット4: ねじれ（交差）チェック
                        # 【修正】end_link（下流端の中点）を基準にする
                        mx = (A[f"x1_{end_link}"] + B[f"x1_{end_link}"]) / 2.0
                        my = (A[f"y1_{end_link}"] + B[f"y1_{end_link}"]) / 2.0

                        # 各リンクでのねじれ条件
                        twisted = False
                        for idx in range(start_layer, end_link):
                            val_1 = np.hypot(A[f"x1_{idx}"] - mx, A[f"y1_{idx}"] - my) - np.hypot(B[f"x1_{idx}"] - mx, B[f"y1_{idx}"] - my)
                            val_0 = np.hypot(A[f"x0_{idx}"] - mx, A[f"y0_{idx}"] - my) - np.hypot(B[f"x0_{idx}"] - mx, B[f"y0_{idx}"] - my)
                            if val_1 * val_0 < 0:
                                twisted = True
                                break
                        if twisted:
                            continue

                        seen_pairs.add(p_key)

                        # =========================================================
                        # ③ γ線物理量の算出
                        # =========================================================
                        track0_p = A["p"]
                        track1_p = B["p"]
                        track0_p_e = A["p_e"]
                        track1_p_e = B["p_e"]

                        track0_ax, track0_ay = A["track_up_ax"], A["track_up_ay"]
                        track1_ax, track1_ay = B["track_up_ax"], B["track_up_ay"]

                        # 運動量・角度の合成
                        gamma_p = track0_p + track1_p
                        gamma_p_e = np.sqrt((track0_p_e ** 2) + (track1_p_e ** 2)) 

                        gamma_Px = A["px"] + B["px"]
                        gamma_Py = A["py"] + B["py"]
                        gamma_P_xy = np.sqrt((gamma_Px ** 2) + (gamma_Py ** 2)) # ここ２乗平均ではなく平均では？

                        gamma_Px_e = np.sqrt((A["px_e"] ** 2) + (B["px_e"] ** 2))
                        gamma_Py_e = np.sqrt((A["py_e"] ** 2) + (B["py_e"] ** 2))
                        gamma_P_xy_e = np.sqrt(((gamma_Px_e**2 * gamma_Px**2) + (gamma_Py_e**2 * gamma_Py**2)) / (gamma_Px**2 + gamma_Py**2)) #ここも改変必要？

                        gamma_ax = (track0_p * track0_ax + track1_p * track1_ax) / gamma_p #運動量に重みをつけた角度
                        gamma_ay = (track0_p * track0_ay + track1_p * track1_ay) / gamma_p

                        # 誤差の伝播計算
                        dXdP0 = (track0_ax - track1_ax) * track1_p / (gamma_p ** 2)
                        dXdP1 = (track1_ax - track0_ax) * track0_p / (gamma_p ** 2)
                        dYdP0 = (track0_ay - track1_ay) * track1_p / (gamma_p ** 2)
                        dYdP1 = (track1_ay - track0_ay) * track0_p / (gamma_p ** 2)
                        
                        gamma_ax_e = np.sqrt((dXdP0**2 * track0_p_e**2) + (dXdP1**2 * track1_p_e**2))
                        gamma_ay_e = np.sqrt((dYdP0**2 * track0_p_e**2) + (dYdP1**2 * track1_p_e**2))

                        # 開き角 (Opening Angle)
                        open_angle_x = np.abs(track0_ax - track1_ax)
                        open_angle_y = np.abs(track0_ay - track1_ay)
                        open_angle = np.sqrt((track0_ax - track1_ax)**2 + (track0_ay - track1_ay)**2)

                        # =========================================================
                        # ④ イベント辞書の構築
                        # =========================================================
                        event = {}
                        for k, v in A.items():
                            event[k] = v
                        for k, v in B.items():
                            event[f"{k}1"] = v

                        # 特有の物理計算結果を上書き・追加
                        event.update({
                            "RMSx_0": A["RMSx"], "RMSy_0": A["RMSy"],
                            "RMSx_1": B["RMSx"], "RMSy_1": B["RMSy"],
                            "RMS_0": A["RMS"], "RMS_0_e": A["RMS_e"],
                            "RMS_1": B["RMS"], "RMS_1_e": B["RMS_e"],

                            "Px_0": A["px"], "Py_0": A["py"],
                            "Px_1": B["px"], "Py_1": B["py"],
                            "track0_p": track0_p, "track1_p": track1_p, "gamma_p": gamma_p,
                            "P_xy_0": A["p_xy"], "P_xy_1": B["p_xy"], "gamma_P_xy": gamma_P_xy,

                            "Px_0_e": A["px_e"], "Py_0_e": A["py_e"],
                            "Px_1_e": B["px_e"], "Py_1_e": B["py_e"],
                            "track0_p_e": track0_p_e, "track1_p_e": track1_p_e, "gamma_p_e": gamma_p_e,
                            "P_xy_0_e": A["p_xy_e"], "P_xy_1_e": B["p_xy_e"], "gamma_P_xy_e": gamma_P_xy_e,

                            "open_angle": open_angle, "open_angle_x": open_angle_x, "open_angle_y": open_angle_y,

                            "gamma_ax": gamma_ax, "gamma_ay": gamma_ay,
                            "gamma_ax_e": gamma_ax_e, "gamma_ay_e": gamma_ay_e,

                            "track0_ax": track0_ax, "track0_ay": track0_ay, "track1_ax": track1_ax, "track1_ay": track1_ay
                        })
                        
                        gamma_events.append(event)

    print("\n\n=== 終了 ===")
    n_events = len(gamma_events)
    n_events_err = np.sqrt(n_events)
    print(f"抽出されたγ線イベント数: {n_events} ± {n_events_err:.1f}")

    # 5. 結果をCSVに書き出し
    df_out = pd.DataFrame(gamma_events)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Saved gamma events to: {output_csv_path}")
