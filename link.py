import pandas as pd
import numpy as np
from linklet_analysis.root import root_to_df
from linklet_analysis.cuts import (CUT_REGISTRY, apply_cut)

# =========================================================
# build_links：最初の layer だけ cut を適用（重複排除なし）
# =========================================================
def build_links(downlayer, n_layers, cut_index=0):
    links = None

    cut = CUT_REGISTRY[cut_index]
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


# =========================================================
# veto_link：指定した次のレイヤーに繋がらない飛跡を抽出する（重複排除なし）
# =========================================================
def veto_link(df_tracks, downlayer, n_layers, cut_index=0):

    veto_fname = f"l-{(n_layers+downlayer-1):03d}-{(n_layers+downlayer):03d}.root"
    print(f"\n===== Veto processing with: {veto_fname} =====")

    df_veto = root_to_df(f"raw_data/{veto_fname}", branches=["id0"])

    veto_ids = df_veto["id0"]
    veto_mask = ~df_tracks[f"id{n_layers+downlayer-1}"].isin(veto_ids)
    
    veto_df = df_tracks[veto_mask]

    print(f"Veto tracks: {len(veto_df)}")
    return veto_df
