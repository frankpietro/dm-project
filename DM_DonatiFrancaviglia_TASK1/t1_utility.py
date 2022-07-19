import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

def logbins(df):
    return np.ceil(1 + (10 / 3) * np.log10(len(df))).astype(int)


def hist(df, col, bins, title, perc=0):
    df = df.sort_values(col)
    size = df[col].size
    cut = int(size*perc)
    plt.figure(figsize=(4, 2))
    if cut:
        sns.histplot(df[col][cut:-cut], bins=bins)
    else:
        sns.histplot(df[col], bins=bins)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()
            

def loghist(df, col, title, perc=0):
    hist(df, col, logbins(df[col]), title, perc)


def double_loghist(df1, df2, col, title, perc=0):
    plt.figure(figsize=(9, 4))
    loghist(df1, col, title, perc)
    loghist(df2, col, title, perc)


def dropextr(series):
    return series.sort_values()
    # [round(len(serie)/20):-round(len(serie)/20)]


def rankings(series, type):
    if type == 0:  # natural binning
        return pd.cut(dropextr(series), bins=logbins(series), labels=range(logbins(series)))
    elif type == 1:  # frequency binning
        return pd.qcut(dropextr(series), q=logbins(series), labels=range(logbins(series)))