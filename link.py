import pandas as pd
import numpy as np
from linklet_analysis.root import root_to_df
from linklet_analysis.cuts import  apply_cut
from linklet_analysis.config import CUT_REGISTRY_QST2021
# =========================================================
# build_links：最初の layer だけ cut を適用（重複排除なし）
# =========================================================
def build_links_old(downlayer, n_layers, cut_index=0):
    links = None

    cut = CUT_REGISTRY_QST2021[cut_index]
    cut_rule = cut["rule"]
    print(f"===== Using cut: {cut['name']} =====")

    for i in range(downlayer, n_layers+downlayer-1):
        fname = f"l-{i:03d}-{i+1:03d}.root"
        print(f"Processing file: {fname}")
        
        df = root_to_df(f"raw_data/{fname}", branches=["id0","id1","x0","x1","y0","y1","z0","z1","ax0","ax1","ay0","ay1","ph0","ph1","dx","dy"])
        if i == 1: #i層目のみcut適応
            df = apply_cut(df, cut_rule)

        df = df.rename(columns={
            "id0": f"id{i}",
            "id1": f"id{i+1}",
            "x0":  f"x0_{i}",
            "x1":  f"x1_{i}",
            "y0":  f"y0_{i}",
            "y1":  f"y1_{i}",
            "z0":  f"z0_{i}",
            "z1":  f"z1_{i}",
            "ax0": f"ax0_{i}",
            "ax1": f"ax1_{i}",
            "ay0": f"ay0_{i}",
            "ay1": f"ay1_{i}",
            "ph0": f"ph0_{i}",
            "ph1": f"ph1_{i}",
            "dx":  f"dx_{i}",
            "dy":  f"dy_{i}"
        })

        if links is None:
            links = df
        else:
            links = pd.merge(links, df, on=f"id{i}", how="inner")

        print(f"Link Tracks 1 to {i+1} = {len(links)}")
        
    return links, cut

# KAKEN（001が上流：small_is_up=True）QST（030が上流：small_is_up=False）
# 上流から下流へ（up_to_down=True）：KAKENなら1,2,3 QSTなら3,2,1
# 下流から上流へ（up_to_down=False）：KAKENなら3,2,1 QSTなら1,2,3
# つまり(True & True) または (False & False) のとき昇順
# (True & False) または (False & True) のとき降順
def build_links(filepath, start_layer, n_layers, cut_rule, small_is_up=True, up_to_down=True):
    links = None

    # 両方のフラグが一致していれば昇順(1)、異なっていれば降順(-1)
    if small_is_up == up_to_down:
        step = 1
    else:
        step = -1

    # フラグの組み合わせによって、実際のループの開始点(start)と終了点(end)を決める
    if up_to_down:
        # start_layerから出発して進むパターン
        start = start_layer
        end = start_layer + (step * n_layers)
    else:
        # end_layerから出発してstart_layerへ向かうパターン
        start = start_layer - (step * (n_layers - 1))
        end = start_layer + step

    print(f"Calculated Loop: start={start}, end={end}, step={step}")

    for i in range(start, end - step, step):
        # 昇順なら (i, i+1)、降順なら (i, i-1) のペアを作り
        # ファイル名の順序を常に「l-{小さい番号}-{大きい番号}」に統一する
        next_i = i + step
        low = min(i, next_i)
        high = max(i, next_i)
        filename = f"l-{low:03d}-{high:03d}.root"
        print(f"Processing file: {filepath}{filename}")

        df = root_to_df(f"{filepath}{filename}", branches=["id0", "id1","x0","x1","y0","y1","z0","z1","ax0","ax1","ay0","ay1","ph0","ph1", "dx", "dy"])

        if (low == start_layer and up_to_down) or (high == start_layer and not(up_to_down)): 
            df = apply_cut(df, cut_rule)

        current_id_suffix = "0" if step == 1 else "1"
        next_id_suffix    = "1" if step == 1 else "0"

        # 重複削除は「現在のレイヤー（結合キー）」を基準に行う
        df = df.drop_duplicates(subset=[f"id{current_id_suffix}", f"id{next_id_suffix}"])
        
        rename_dict = {
            f"id{current_id_suffix}": f"id{i}",
            f"id{next_id_suffix}":    f"id{next_i}",
            
            # 一律で `_i` を付与
            "x0":  f"x0_{low}",
            "x1":  f"x1_{low}",
            "y0":  f"y0_{low}",
            "y1":  f"y1_{low}",
            "z0":  f"z0_{low}",
            "z1":  f"z1_{low}",
            "ax0": f"ax0_{low}",
            "ax1": f"ax1_{low}",
            "ay0": f"ay0_{low}",
            "ay1": f"ay1_{low}",
            "ph0": f"ph0_{low}",
            "ph1": f"ph1_{low}",
            "dx":  f"dx_{low}",
            "dy":  f"dy_{low}"
        }
        # # 動的にリネーム用の辞書を作成
        # rename_dict = {
        #     f"id{current_id_suffix}": f"id{i}",
        #     f"id{next_id_suffix}":    f"id{next_i}",
        # }
        # # その他の物理量も、ファイル内での「0(小さい方)」「1(大きい方)」に合わせてリネーム
        # for sfx in ["0", "1"]:
        #     # low/highのどちらに属するレイヤーかを判定
        #     layer_num = low if sfx == "0" else high
        #     for var in ["x", "y", "z", "ax", "ay", "ph"]:
        #         rename_dict[f"{var}{sfx}"] = f"{var}{sfx}_{layer_num}"

        df = df.rename(columns=rename_dict)
        # ---------------------------------------------------------

        if links is None:
            links = df
        else:
            # 常に「現在のレイヤー番号(i)」をキーにしてインナージョイン
            links = pd.merge(links, df, on=f"id{i}", how="inner")

        print(f"Link Tracks {start} to {next_i} = {len(links)}")


    # #  出力されるカラムの並び順を一律「レイヤー番号の昇順」に強制ソート
    # if links is not None:
    #     # ID系のカラムを id3, id4... のように数字の昇順に並べる
    #     id_cols = sorted([c for c in links.columns if c.startswith("id")], key=lambda x: int(x.replace("id", "")))
        
    #     # 物理量系のカラムを 末尾の _3, _4... の昇順に並べる
    #     other_cols = [c for c in links.columns if not c.startswith("id")]
    #     other_cols_sorted = sorted(other_cols, key=lambda x: int(x.split("_")[-1]))
        
    #     # 合体させて並び替え
    #     links = links[id_cols + other_cols_sorted]

    return links


# =========================================================
# veto_link：指定した次のレイヤーに繋がらない飛跡を抽出する（重複排除なし）
# =========================================================
def veto_link(filepath, df_tracks, start_layer, veto_layer, cut_rule=0):

    veto_fname = f"l-{(start_layer):03d}-{(veto_layer):03d}.root"
    print(f"\n===== Veto processing with: {veto_fname} =====")

    df_veto = root_to_df(f"{filepath}{veto_fname}", branches=["id0"])

    veto_ids = df_veto["id0"]
    veto_mask = ~df_tracks[f"id{start_layer}"].isin(veto_ids)
    
    veto_df = df_tracks[veto_mask]

    print(f"Veto tracks: {len(veto_df)}")
    return veto_df