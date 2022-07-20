import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import itertools
from sklearn.model_selection import RandomizedSearchCV

def cross_validation(model, x, y, n_splits):
    """Return validation scores across the k folds of cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    val_score = []
    for train_index, test_index in skf.split(x, y):
        model.fit(x[train_index], y[train_index])
        val_score.append(model.score(x[test_index], y[test_index].ravel()))
    return np.array(val_score)

def cross_validation_summary(model, x, y):
    """Returns validation accuracy score of model (mean and std over all the splits)."""
    val_score = cross_validation(model, x, y, 5)
    return val_score.mean(), val_score.std()
