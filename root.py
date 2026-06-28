import ROOT
import pandas as pd

# ======================
# ROOT → pandas 変換
# ======================
def root_to_df(filename, tree="nt", branches=None):
    df = ROOT.RDataFrame(tree, filename)
    if branches is None:
        cols = list(df.GetColumnNames())
    else:
        cols = branches

    data = df.AsNumpy(cols)
    return pd.DataFrame(data)
