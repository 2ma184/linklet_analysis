import pandas as pd
from linklet_analysis.root import root_to_df
from linklet_analysis.cuts import  apply_cut

# build_links：すべての layer で cut を適用（重複排除なし）

# KAKEN（001が上流：small_is_up=True）QST（030が上流：small_is_up=False）
# 上流から下流へ（up_to_down=True）：KAKENなら1,2,3 QSTなら3,2,1
# 下流から上流へ（up_to_down=False）：KAKENなら3,2,1 QSTなら1,2,3
# つまり(True & True) または (False & False) のとき昇順
# (True & False) または (False & True) のとき降順
def build_links(filepath, start_layer, n_layers, cut_rule, scan_direction=True, up_to_down=True):
    df_links = None

    # 両方のフラグが一致していれば昇順(1)、異なっていれば降順(-1)
    if scan_direction == up_to_down:
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

        # if (low == start_layer and up_to_down) or (high == start_layer and not(up_to_down)): 
        df = apply_cut(df, cut_rule)

        current_id_suffix = "0" if step == 1 else "1"
        next_id_suffix    = "1" if step == 1 else "0"

        # 重複削除は「現在のレイヤー（結合キー）」を基準に行う
        df = df.drop_duplicates(subset=[f"id{current_id_suffix}", f"id{next_id_suffix}"])
        
        rename_dict = {
            f"id{current_id_suffix}": f"id{i}",
            f"id{next_id_suffix}":    f"id{next_i}",
            
            # linkletを見分けれるようにl-001-002なら12を付与
            "x0":  f"x0_{low}{high}",
            "x1":  f"x1_{low}{high}",
            "y0":  f"y0_{low}{high}",
            "y1":  f"y1_{low}{high}",
            "z0":  f"z0_{low}{high}",
            "z1":  f"z1_{low}{high}",
            "ax0": f"ax0_{low}{high}",
            "ax1": f"ax1_{low}{high}",
            "ay0": f"ay0_{low}{high}",
            "ay1": f"ay1_{low}{high}",
            "ph0": f"ph0_{low}{high}",
            "ph1": f"ph1_{low}{high}",
            "dx":  f"dx_{low}{high}",
            "dy":  f"dy_{low}{high}"
        }
        df = df.rename(columns=rename_dict)

        # ---------------------------------------------------------

        if df_links is None:
            df_links = df
        else:
            # 常に「現在のレイヤー番号(i)」をキーにしてインナージョイン
            df_links = pd.merge(df_links, df, on=f"id{i}", how="inner")

        print(f"Link Tracks {start} to {next_i} = {len(df_links)}")


    #  出力されるカラムの並び順を一律「レイヤー番号の昇順」に強制ソート
    if df_links is not None:
        # ID系のカラムを id3, id4... のように数字の昇順に並べる
        id_cols = sorted([c for c in df_links.columns if c.startswith("id")], key=lambda x: int(x.replace("id", "")))
        
        # 物理量系のカラムを 末尾の _3, _4... の昇順に並べる
        other_cols = [c for c in df_links.columns if not c.startswith("id")]
        other_cols_sorted = sorted(other_cols, key=lambda x: int(x.split("_")[-1]))
        
        # 合体させて並び替え
        df_links = df_links[id_cols + other_cols_sorted]

    return df_links


# =========================================================
# veto_link：指定した次のレイヤーに繋がらない飛跡を抽出する（重複排除なし）
# =========================================================
def veto_link(filepath, df_tracks, start_layer, veto_layer, cut_rule=0):

    veto_fname = f"l-{(start_layer):03d}-{(veto_layer):03d}.root"
    print(f"Veto processing with: {veto_fname}")

    df_veto = root_to_df(f"{filepath}{veto_fname}", branches=["id0"])

    veto_ids = df_veto["id0"]
    veto_mask = ~df_tracks[f"id{start_layer}"].isin(veto_ids)
    
    df_veto_tracks = df_tracks[veto_mask]

    print(f"Veto tracks: {len(df_veto_tracks)}")
    return df_veto_tracks