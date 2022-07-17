import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_df(df, method):
    if method == 1:
        array = StandardScaler().fit_transform(df)
    else:
        array = MinMaxScaler().fit_transform(df)
    return array, pd.DataFrame(data=array, index=df.index, columns=df.columns)


def rankings(series, type, bins):
    if type == 0:  # natural binning
        return pd.cut(series.sort_values(), bins=bins, labels=range(bins))
    elif type == 1:  # frequency binning
        return pd.qcut(series.sort_values(), q=bins, labels=range(bins))