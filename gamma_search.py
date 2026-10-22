import pandas as pd
import numpy as np

from scipy.spatial import cKDTree

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
        nan3 = np.full(3, np.nan)
        return np.nan, nan3.copy(), nan3.copy(), nan3.copy()
    
    #二つの直線の最近接点でのs,t
    s = (b*e - c*d) / det_nom 
    t = (a*e - b*d) / det_nom

    C_A = P_A + s*V_A
    C_B = P_B + t*V_B

    vertex = 0.5*(C_A + C_B)
    closest_distance = np.linalg.norm(C_A - C_B)

    return closest_distance, vertex, C_A, C_B


def extract_gamma_events(df, start_layer, n_layers):
    #例は全て start_layer=5, n_layer=5の時
    end_layer = start_layer - n_layers + 1 # 例: 5 - 5 + 1 = 1
    start_link_name = f"{start_layer-1}{start_layer}"

    # DataFrameを辞書リストへ変換
    tracks = df.to_dict(orient="records")
    if len(tracks) < 2:
        return pd.DataFrame()

    # =========================================================
    # KDTreeによる近接trackペアの候補抽出
    # =========================================================
    # 全trackについて、最上流layerの下流側基準面における (x1, y1) 座標を2次元配列として取得する。
    points = np.array([[t[f"x1_{start_link_name}"], t[f"y1_{start_link_name}"]] for t in tracks ], dtype=float)

    # (x1, y1) 平面上にKDTreeを構築する
    tree = cKDTree(points)
    
    # 計算量削減のため互いにある距離以内にあるtrackの組だけを探索する
    SEARCH_RADIUS = 1000.0   # 最上流layerのz1面において、候補として比較する最大距離 [µm]

    # SEARCH_RADIUS以内にあるtrackのインデックスペア(i, j)を取得する
    # query_pairsはi < jの組だけを返すため、(A, B)と(B, A)を重複して調べることはない。    
    candidate_pairs = tree.query_pairs(r=SEARCH_RADIUS, output_type="ndarray")  # [[i0, j0], [i1, j1], ...] のNumPy配列として返される

    n_candidates = len(candidate_pairs)

    seen_pairs = set()
    gamma_events = []

    print(f"Starting KDTree pair search (Link {start_link_name}, radius={SEARCH_RADIUS} µm)")
    print(f"Candidate pairs: {n_candidates}")

    for pair_count, (i, j) in enumerate(candidate_pairs, start=1):
        A = tracks[int(i)]
        B = tracks[int(j)]
        if pair_count % 1000 == 0 or pair_count == n_candidates:
            print(f"\rChecked pairs: {pair_count}", end="", flush=True)

        # =========================================================
        # IDベースの重複排除
        # ========================================================
        a_id = int(A[f"id{start_layer}"])
        b_id = int(B[f"id{start_layer}"])

        if a_id == b_id: #最上流で同じidなら飛ばす(open_angleが定義できない)
            continue
            
        pair_key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        if pair_key in seen_pairs:
            continue #見たことあるidなら飛ばす

        # =========================================================
        #  γ線の物理条件カット（幾何学的収束・ねじれ判定）
        # =========================================================
        # まずは様々な値を計算
        # ----- dtc : distance track center (track同士の距離) ------
        dtc_0 = []
        dtc_1 = []
        for idx in range(end_layer, start_layer): # 例: start_layer=5, n_layers=5 のとき、idx は 1, 2, 3, 4 と動く
            dtc_0.append(np.hypot(A[f"x0_{idx}{idx+1}"] - B[f"x0_{idx}{idx+1}"], A[f"y0_{idx}{idx+1}"] - B[f"y0_{idx}{idx+1}"])) 
            dtc_1.append(np.hypot(A[f"x1_{idx}{idx+1}"] - B[f"x1_{idx}{idx+1}"], A[f"y1_{idx}{idx+1}"] - B[f"y1_{idx}{idx+1}"])) # 配列の末尾 ([-1]) が 最上流 (idx=4,x1_45) のデータになる

        
        # ----- Vertex (最近接点) ------
        start_layer_pointA = np.array([A[f"x1_{start_link_name}"], A[f"y1_{start_link_name}"], A[f"z1_{start_link_name}"],], dtype=float)
        start_layer_pointB = np.array([B[f"x1_{start_link_name}"], B[f"y1_{start_link_name}"], B[f"z1_{start_link_name}"],], dtype=float)

        start_layer_vectorA = np.array([A["ux"], A["uy"], A["uz"]], dtype=float)
        start_layer_vectorB = np.array([B["ux"], B["uy"], B["uz"]], dtype=float)

        track_distance0, vertex, C_A, C_B = calc_vertex(start_layer_pointA, start_layer_vectorA, start_layer_pointB, start_layer_vectorB)

        # 平行または計算不能な組を除外
        if (not np.isfinite(track_distance0) or not np.all(np.isfinite(vertex)) ):
            continue
        
        # vertex_dz = z1からvertexまでの上流方向距離
        z1_mean = 0.5 * (start_layer_pointA[2] + start_layer_pointB[2] )
        vertex_dz = (-(vertex[2] - z1_mean) ) #zが負の時は-がかかる
        

        # -- カット1: vertexがz1より上流側にある
        C_A_is_upstream = (-(C_A[2] - start_layer_pointA[2]) > 0 ) #zが負の時は-がかかる
        C_B_is_upstream = (-(C_B[2] - start_layer_pointB[2]) > 0 ) #zが負の時は-がかかる
        if not (C_A_is_upstream and C_B_is_upstream):
            continue

        # -- カット: vertexが次の乳剤層に相当する範囲にある
        if not (200.0 < vertex_dz < 350.0):
            continue

        # -- カット2: vertexでの最近接距離が5 µm未満
        if track_distance0 >= 5.0:
            continue

        # -- カット3: 上流（配列の末尾）に向かって2本の距離が狭まっていること
        # 上流から下流へ、値が「単調増増加」しているかチェック (dtc[idx] > dt_next)
        if not all(dtc_0[idx] > dt0_next for idx, dt0_next in enumerate(dtc_0[1:])):
            continue
        if not all(dtc_1[idx] > dt1_next for idx, dt1_next in enumerate(dtc_1[1:])):
            continue
        if not (dtc_1[-1] < dtc_0[-1] < dtc_0[-2]): # 根本付近の厳密な順序チェック,dct_1_45 < dtc_0_45 < dtc_0_34
            continue

        # -- カット4: ねじれ（交差）チェック
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
        #  γ線物理量の算出
        # =========================================================
        # trackの運動量ベクトルの読み込み
        trackA_p = A["p"]
        trackB_p = B["p"]
        trackA_p_e = A["p_e"]
        trackB_p_e = B["p_e"]

        ## -単位方向ベクトル
        uxA, uyA, uzA = A["ux"], A["uy"], A["uz"]
        uxB, uyB, uzB = B["ux"], B["uy"], B["uz"]

        ## -運動量ベクトル
        trackA_Px, trackA_Py, trackA_Pz = A["Px"], A["Py"], A["Pz"]
        trackB_Px, trackB_Py, trackB_Pz = B["Px"], B["Py"], B["Pz"]

        ## -エネルギー
        trackA_E, trackA_E_e = A["E"], A["E_e"]
        trackB_E, trackB_E_e = B["E"], B["E_e"]

        # γ線の運動量ベクトルへ合成（原子核の反跳は無視）
        gamma_Px = trackA_Px + trackB_Px
        gamma_Py = trackA_Py + trackB_Py
        gamma_Pz = trackA_Pz + trackB_Pz
        gamma_p = np.sqrt(gamma_Px**2 + gamma_Py**2 + gamma_Pz**2)

        # γ線の傾き
        gamma_ax = gamma_Px / gamma_Pz
        gamma_ay = gamma_Py / gamma_Pz

        # 運動量誤差のみを伝播した暫定的なγ線傾き誤差
        # トラック方向そのものの測定誤差は未考慮
        dgamma_ax_dpA = (uxA * gamma_Pz - gamma_Px * uzA) / gamma_Pz**2
        dgamma_ax_dpB = (uxB * gamma_Pz - gamma_Px * uzB) / gamma_Pz**2
        gamma_ax_e = np.sqrt((dgamma_ax_dpA * trackA_p_e)**2 + (dgamma_ax_dpB * trackB_p_e)**2)

        dgamma_ay_dpA = (uyA * gamma_Pz - gamma_Py * uzA) / gamma_Pz**2
        dgamma_ay_dpB = (uyB * gamma_Pz - gamma_Py * uzB) / gamma_Pz**2
        gamma_ay_e = np.sqrt((dgamma_ay_dpA * trackA_p_e)**2+ (dgamma_ay_dpB * trackB_p_e)**2)

        # γ線のエネルギー
        gamma_E = trackA_E + trackB_E
        gamma_E_e = np.sqrt(trackA_E_e**2 + trackB_E_e**2)

        # 誤差計算
        dp_gamma_dpA = (gamma_Px * uxA + gamma_Py * uyA + gamma_Pz * uzA ) / gamma_p
        dp_gamma_dpB = (gamma_Px * uxB + gamma_Py * uyB + gamma_Pz * uzB ) / gamma_p
        gamma_p_e = np.sqrt((dp_gamma_dpA * trackA_p_e)**2 + (dp_gamma_dpB * trackB_p_e)**2)

        # 開き角 (Opening Angle) [rad]
        cos_open_angle = (uxA * uxB + uyA * uyB + uzA * uzB)
        cos_open_angle = np.clip(cos_open_angle, -1.0, 1.0)# 浮動小数点誤差で±1を超えないようにする
        open_angle = np.arccos(cos_open_angle)

        # vertex
        vertex_x = vertex[0]
        vertex_y = vertex[1]
        vertex_z = vertex[2]

        #gamma線のpair同士の距離
        track_distance1 = dtc_1[-1]
        track_distance2 = dtc_1[-2]
        track_distance3 = dtc_1[-3]
        track_distance4 = dtc_1[-4]

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
            "gamma_p": gamma_p,
            "gamma_p_e": gamma_p_e,

            "gamma_Px": gamma_Px,
            "gamma_Py": gamma_Py,
            "gamma_Pz": gamma_Pz,
            "gamma_Pt": np.hypot(gamma_Px, gamma_Py),

            "gamma_E": gamma_E,
            "gamma_E_e": gamma_E_e,

            "gamma_ax": gamma_ax, "gamma_ay": gamma_ay,
            "gamma_ax_e": gamma_ax_e, "gamma_ay_e": gamma_ay_e,

            "open_angle": open_angle,

            "track_distance1": track_distance1, "track_distance2": track_distance2, "track_distance3": track_distance3, "track_distance4": track_distance4,

            "track_distance0": track_distance0, "vertex_x": vertex_x, "vertex_y": vertex_y, "vertex_z": vertex_z, "vertex_dz": vertex_dz
        })

        gamma_events.append(event)

    print("\n=== 終了 ===")
    n_events = len(gamma_events)
    n_events_err = np.sqrt(n_events)
    print(f"抽出されたγ線イベント数: {n_events} ± {n_events_err:.1f}")

    return pd.DataFrame(gamma_events)


# 過去の参考
#================================================
# # 運動量・角度の合成
# gamma_p = trackA_p + trackB_p
# gamma_p_e = np.sqrt((trackA_p_e ** 2) + (trackB_p_e ** 2)) 

# gamma_Px = A["px"] + B["px"]
# gamma_Py = A["py"] + B["py"]
# gamma_P_xy = np.sqrt((gamma_Px ** 2) + (gamma_Py ** 2)) # ここ２乗平均ではなく平均では？

# gamma_Px_e = np.sqrt((A["px_e"] ** 2) + (B["px_e"] ** 2))
# gamma_Py_e = np.sqrt((A["py_e"] ** 2) + (B["py_e"] ** 2))
# gamma_P_xy_e = np.sqrt(((gamma_Px_e**2 * gamma_Px**2) + (gamma_Py_e**2 * gamma_Py**2)) / (gamma_Px**2 + gamma_Py**2)) #ここも改変必要？

# gamma_ax = (trackA_p * trackA_ax0 + trackB_p * trackB_ax0) / gamma_p #運動量に重みをつけた角度
# gamma_ay = (trackA_p * trackA_ay0 + trackB_p * trackB_ay0) / gamma_p

# # 誤差の伝播計算
# dXdP_A = (trackA_ax0 - trackB_ax0) * trackB_p / (gamma_p ** 2)
# dXdP_B = (trackB_ax0 - trackA_ax0) * trackA_p / (gamma_p ** 2)
# dYdP_A = (trackA_ay0 - trackB_ay0) * trackB_p / (gamma_p ** 2)
# dYdP_B = (trackB_ay0 - trackA_ay0) * trackA_p / (gamma_p ** 2)

# gamma_ax_e = np.sqrt((dXdP_A**2 * trackA_p_e**2) + (dXdP_B**2 * trackB_p_e**2))
# gamma_ay_e = np.sqrt((dYdP_A**2 * trackA_p_e**2) + (dYdP_B**2 * trackB_p_e**2))
#=================================================


def shift_track_xy(track, dx, dy, end_layer, start_layer):
    shifted = track.copy()

    for idx in range(end_layer, start_layer):
        link = f"{idx}{idx+1}"

        shifted[f"x0_{link}"] += dx
        shifted[f"x1_{link}"] += dx
        shifted[f"y0_{link}"] += dy
        shifted[f"y1_{link}"] += dy

    return shifted

def extract_gamma_events_chance(df, start_layer, n_layers,shift_xy=(0.0, 0.0)):
    #例は全て start_layer=5, n_layer=5の時
    end_layer = start_layer - n_layers + 1 # 例: 5 - 5 + 1 = 1
    start_link_name = f"{start_layer-1}{start_layer}"

    dx, dy = shift_xy

    # DataFrameを辞書リストへ変換
    tracks = df.to_dict(orient="records")
    if len(tracks) < 2:
        return pd.DataFrame()

    # =========================================================
    # KDTreeによる近接trackペアの候補抽出：パターンA
    # =========================================================
    # 全trackについて、最上流layerの下流側基準面における (x1, y1) 座標を2次元配列として取得する。
    points = np.array([[t[f"x1_{start_link_name}"], t[f"y1_{start_link_name}"]] for t in tracks ], dtype=float)

    # (x1, y1) 平面上にKDTreeを構築する
    tree = cKDTree(points)
    
    # 計算量削減のため互いにある距離以内にあるtrackの組だけを探索する
    SEARCH_RADIUS = 1000.0   # 最上流layerのz1面において、候補として比較する最大距離 [µm]

    # SEARCH_RADIUS以内にあるtrackのインデックスペア(i, j)を取得する
    # query_pairsはi < jの組だけを返すため、(A, B)と(B, A)を重複して調べることはない。    
    candidate_pairs = tree.query_pairs(r=SEARCH_RADIUS, output_type="ndarray")  # [[i0, j0], [i1, j1], ...] のNumPy配列として返される

    # これは以下の式にほぼ同じ
    # candidate_pairs = []
    # for i in range(len(points)):
    #     for j in range(i + 1, len(points)):
    #         distance = np.sqrt((points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2 )
    #         if distance <= SEARCH_RADIUS:
    #             candidate_pairs.append((i, j))


    # =========================================================
    # KDTreeによる近接trackペアの候補抽出：パターンA
    # =========================================================
    # points = np.array([[t[f"x1_{start_link_name}"], t[f"y1_{start_link_name}"]] for t in tracks ], dtype=float)
    # shifted_points = points + np.array([dx, dy]) # B側だけを平行移動

    # tree_A = cKDTree(points)
    # tree_B = cKDTree(shifted_points)
    
    # SEARCH_RADIUS = 100.0   
    # # SEARCH_RADIUS以内にあるtrackのAとBのインデックスペア(i, j)を取得する
    # neighbors = tree_A.query_ball_tree(tree_B, r=SEARCH_RADIUS)
    # candidate_pairs = np.asarray([(i, j) for i, js in enumerate(neighbors) for j in js if i != j ], dtype=int ).reshape(-1, 2) # [[i0, j0], [i1, j1], ...] のNumPy配列として返される
    # # [   [2, 5],  # Aの0番に近いBは2番と5番
    # #     [],      # Aの1番に近いBはない
    # #     [0],     # Aの2番に近いBは0番
    # #     [4, 7] ] # Aの3番に近いBは4番と7番

    # # これは以下の式にほぼ同じ
    # # neighbors = []
    # # for i in range(len(points_A)):
    # #     js = []
    # #     for j in range(len(points_B)):
    # #         distance = np.sqrt((points_A[i, 0] - points_B[j, 0])**2 + (points_A[i, 1] - points_B[j, 1])**2)
    # #         if distance <= SEARCH_RADIUS:
    # #             js.append(j)
    # #     neighbors.append(js)






    n_candidates = len(candidate_pairs)

    seen_pairs = set()
    gamma_events = []

    print(f"Starting KDTree pair search (Link {start_link_name}, radius={SEARCH_RADIUS} µm)")
    print(f"Candidate pairs: {n_candidates}")

    for pair_count, (i, j) in enumerate(candidate_pairs, start=1):
        A = tracks[int(i)]
        B = shift_track_xy(tracks[int(j)], dx, dy, end_layer, start_layer)

        if pair_count % 1000 == 0 or pair_count == n_candidates:
            print(f"\rChecked pairs: {pair_count}", end="", flush=True)

        # =========================================================
        # IDベースの重複排除
        # ========================================================
        a_id = int(A[f"id{start_layer}"])
        b_id = int(B[f"id{start_layer}"])
        

        if a_id == b_id: #最上流で同じidなら飛ばす(open_angleが定義できない)
            continue
            
        
        # A-BとB-Aは同じ物理的なペア
        pair_key = tuple(sorted((a_id, b_id)))

        if pair_key in seen_pairs:
            continue

        # =========================================================
        #  γ線の物理条件カット（幾何学的収束・ねじれ判定）
        # =========================================================
        # まずは様々な値を計算
        # ----- dtc : distance track center (track同士の距離) ------
        dtc_0 = []
        dtc_1 = []
        for idx in range(end_layer, start_layer): # 例: start_layer=5, n_layers=5 のとき、idx は 1, 2, 3, 4 と動く
            dtc_0.append(np.hypot(A[f"x0_{idx}{idx+1}"] - B[f"x0_{idx}{idx+1}"], A[f"y0_{idx}{idx+1}"] - B[f"y0_{idx}{idx+1}"])) 
            dtc_1.append(np.hypot(A[f"x1_{idx}{idx+1}"] - B[f"x1_{idx}{idx+1}"], A[f"y1_{idx}{idx+1}"] - B[f"y1_{idx}{idx+1}"])) # 配列の末尾 ([-1]) が 最上流 (idx=4,x1_45) のデータになる

        
        # ----- Vertex (最近接点) ------
        start_layer_pointA = np.array([A[f"x1_{start_link_name}"], A[f"y1_{start_link_name}"], A[f"z1_{start_link_name}"],], dtype=float)
        start_layer_pointB = np.array([B[f"x1_{start_link_name}"], B[f"y1_{start_link_name}"], B[f"z1_{start_link_name}"],], dtype=float)

        start_layer_vectorA = np.array([A["ux"], A["uy"], A["uz"]], dtype=float)
        start_layer_vectorB = np.array([B["ux"], B["uy"], B["uz"]], dtype=float)

        track_distance0, vertex, C_A, C_B = calc_vertex(start_layer_pointA, start_layer_vectorA, start_layer_pointB, start_layer_vectorB)

        # 平行または計算不能な組を除外
        if (not np.isfinite(track_distance0) or not np.all(np.isfinite(vertex)) ):
            continue
        
        # vertex_dz = z1からvertexまでの上流方向距離
        z1_mean = 0.5 * (start_layer_pointA[2] + start_layer_pointB[2] )
        vertex_dz = (-(vertex[2] - z1_mean) ) #zが負の時は-がかかる

        
        # ==================　γ線カット　===============================
        # -- カット: vertexがz1より上流側にある
        C_A_is_upstream = (-(C_A[2] - start_layer_pointA[2]) > 0 ) #zが負の時は-がかかる
        C_B_is_upstream = (-(C_B[2] - start_layer_pointB[2]) > 0 ) #zが負の時は-がかかる
        if not (C_A_is_upstream and C_B_is_upstream):
            continue

        # -- カット: vertexが次の乳剤層に相当する範囲にある
        if not (200.0 < vertex_dz < 350.0):
            continue

        # -- カット: vertexでの最近接距離が5 µm未満
        if track_distance0 >= 5.0:
            continue

        # -- カット: 上流（配列の末尾）に向かって2本の距離が狭まっていること
        # 上流から下流へ、値が「単調増増加」しているかチェック (dtc[idx] > dt_next)
        if not all(dtc_0[idx] > dt0_next for idx, dt0_next in enumerate(dtc_0[1:])):
            continue
        if not all(dtc_1[idx] > dt1_next for idx, dt1_next in enumerate(dtc_1[1:])):
            continue
        if not (dtc_1[-1] < dtc_0[-1] < dtc_0[-2]): # 根本付近の厳密な順序チェック,dct_1_45 < dtc_0_45 < dtc_0_34
            continue

        # -- カット: ねじれ（交差）チェック
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

        # =========================================================
        #  γ線物理量の算出
        # =========================================================
        # trackの運動量ベクトルの読み込み
        trackA_p = A["p"]
        trackB_p = B["p"]
        trackA_p_e = A["p_e"]
        trackB_p_e = B["p_e"]

        ## -単位方向ベクトル
        uxA, uyA, uzA = A["ux"], A["uy"], A["uz"]
        uxB, uyB, uzB = B["ux"], B["uy"], B["uz"]

        ## -運動量ベクトル
        trackA_Px, trackA_Py, trackA_Pz = A["Px"], A["Py"], A["Pz"]
        trackB_Px, trackB_Py, trackB_Pz = B["Px"], B["Py"], B["Pz"]

        ## -エネルギー
        trackA_E, trackA_E_e = A["E"], A["E_e"]
        trackB_E, trackB_E_e = B["E"], B["E_e"]

        # γ線の運動量ベクトルへ合成（原子核の反跳は無視）
        gamma_Px = trackA_Px + trackB_Px
        gamma_Py = trackA_Py + trackB_Py
        gamma_Pz = trackA_Pz + trackB_Pz
        gamma_p = np.sqrt(gamma_Px**2 + gamma_Py**2 + gamma_Pz**2)

        # γ線の傾き
        gamma_ax = gamma_Px / gamma_Pz
        gamma_ay = gamma_Py / gamma_Pz

        # 運動量誤差のみを伝播した暫定的なγ線傾き誤差
        # トラック方向そのものの測定誤差は未考慮
        dgamma_ax_dpA = (uxA * gamma_Pz - gamma_Px * uzA) / gamma_Pz**2
        dgamma_ax_dpB = (uxB * gamma_Pz - gamma_Px * uzB) / gamma_Pz**2
        gamma_ax_e = np.sqrt((dgamma_ax_dpA * trackA_p_e)**2 + (dgamma_ax_dpB * trackB_p_e)**2)

        dgamma_ay_dpA = (uyA * gamma_Pz - gamma_Py * uzA) / gamma_Pz**2
        dgamma_ay_dpB = (uyB * gamma_Pz - gamma_Py * uzB) / gamma_Pz**2
        gamma_ay_e = np.sqrt((dgamma_ay_dpA * trackA_p_e)**2+ (dgamma_ay_dpB * trackB_p_e)**2)

        # γ線のエネルギー
        gamma_E = trackA_E + trackB_E
        gamma_E_e = np.sqrt(trackA_E_e**2 + trackB_E_e**2)

        # 誤差計算
        dp_gamma_dpA = (gamma_Px * uxA + gamma_Py * uyA + gamma_Pz * uzA ) / gamma_p
        dp_gamma_dpB = (gamma_Px * uxB + gamma_Py * uyB + gamma_Pz * uzB ) / gamma_p
        gamma_p_e = np.sqrt((dp_gamma_dpA * trackA_p_e)**2 + (dp_gamma_dpB * trackB_p_e)**2)

        # 開き角 (Opening Angle) [rad]
        cos_open_angle = (uxA * uxB + uyA * uyB + uzA * uzB)
        cos_open_angle = np.clip(cos_open_angle, -1.0, 1.0)# 浮動小数点誤差で±1を超えないようにする
        open_angle = np.arccos(cos_open_angle)

        # vertex
        vertex_x = vertex[0]
        vertex_y = vertex[1]
        vertex_z = vertex[2]

        #gamma線のpair同士の距離
        track_distance1 = dtc_1[-1]
        track_distance2 = dtc_1[-2]
        track_distance3 = dtc_1[-3]
        track_distance4 = dtc_1[-4]

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
            "shift_dx": dx,
            "shift_dy": dy,

            "gamma_p": gamma_p,
            "gamma_p_e": gamma_p_e,

            "gamma_Px": gamma_Px,
            "gamma_Py": gamma_Py,
            "gamma_Pz": gamma_Pz,
            "gamma_Pt": np.hypot(gamma_Px, gamma_Py),

            "gamma_E": gamma_E,
            "gamma_E_e": gamma_E_e,

            "gamma_ax": gamma_ax, "gamma_ay": gamma_ay,
            "gamma_ax_e": gamma_ax_e, "gamma_ay_e": gamma_ay_e,

            "open_angle": open_angle,

            "track_distance1": track_distance1, "track_distance2": track_distance2, "track_distance3": track_distance3, "track_distance4": track_distance4,

            "track_distance0": track_distance0, "vertex_x": vertex_x, "vertex_y": vertex_y, "vertex_z": vertex_z, "vertex_dz": vertex_dz
        })

        seen_pairs.add(pair_key)
        gamma_events.append(event)
        

    id_col = f"id{start_layer}"

    print("\n=== 終了 ===")
    n_events = len(gamma_events)
    n_events_err = np.sqrt(n_events)
    print(f"抽出されたγ線イベント数: {n_events} ± {n_events_err:.1f}")

    return pd.DataFrame(gamma_events)