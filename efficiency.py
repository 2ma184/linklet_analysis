from linklet_analysis.root import root_to_df
from linklet_analysis.cuts import apply_cut
import numpy as np
import pandas as pd

# ======================
# efficiency計算
# ======================
def cal_eff(filename1,filename2,cut_rule):
    # root fileの読み込み->dataflameへ
    df_a = root_to_df(filename1, branches=["id0", "id1","x0","x1","y0","y1","z0","z1","ax0","ax1","ay0","ay1","ph0","ph1"]) 
    df_b = root_to_df(filename2, branches=["id0", "id1","x0","x1","y0","y1","z0","z1","ax0","ax1","ay0","ay1","ph0","ph1"])
    # CUTの適応
    df_a_cut = apply_cut(df_a, cut_rule)
    df_b_cut = apply_cut(df_b, cut_rule)
    # 重複削除
    df_a_unique = df_a_cut.drop_duplicates(subset="id0")
    df_b_unique = df_b_cut.drop_duplicates(subset="id0")
    # カラム名のリネーム（id0だけは結合キーとしてそのまま残す）
    df_a_unique = df_a_unique.add_suffix("_a").rename(columns={"id0_a": "id0"})
    df_b_unique = df_b_unique.add_suffix("_b").rename(columns={"id0_b": "id0"})
    # (i,i+1)linkletと(i,i+2)linkletを結合して(i,i+1,i+2)のdfを作る
    coin_df = pd.merge(df_a_unique, df_b_unique, on="id0", how="inner")

    coin_total = len(coin_df)
    b_total = len(df_b_unique)
    print(f"Coincidence tracks:{coin_total}")
    print(f"All tracks :{b_total}")

    eff = 100.0 * coin_total / b_total
    eff_err = 100.0 * np.sqrt((b_total - coin_total) * coin_total) / (b_total * np.sqrt(b_total))
    print(f"Efficiency: {eff:.2f} ± {eff_err:.2f} %\n")

    return eff, eff_err
