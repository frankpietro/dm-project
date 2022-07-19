import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import itertools
from sklearn.model_selection import RandomizedSearchCV


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


def randomized_cv(model, x, y, param_d, n_iter=100):
    """Perform hyper-parameters grid search and return best configuration."""
    rf_random = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_d,
        n_iter = n_iter,
        cv = 5,
        n_jobs = -1
    )

    # Run grid search
    rf_random.fit(x, y)
    mean_acc, std_acc = cross_validation_summary(rf_random.best_estimator_, x, y)

    # Print configuration  and stats about best model
    print(f'{rf_random.best_estimator_}\n mean acc: {mean_acc:.3f}\n std_acc: {std_acc:.3f}')

    return rf_random.best_estimator_