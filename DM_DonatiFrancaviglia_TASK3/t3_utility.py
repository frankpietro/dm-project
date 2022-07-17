import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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