import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def logbins(df):
    return np.ceil(1 + (10 / 3) * np.log10(len(df))).astype(int)

def dropextr(serie):
    return serie.sort_values()
    # [round(len(serie)/20):-round(len(serie)/20)]

def hist(serie, bins, title):
    plt.figure(figsize=(4, 2))
    sns.histplot(serie, bins=bins)
    plt.xticks(rotation=45)
    plt.title(title)
    return plt.show()

def loghist(serie, title):
    return hist(serie, logbins(serie), title)

def rankings(serie):
    return pd.cut(dropextr(serie), bins = logbins(serie), labels = range(logbins(serie)))
