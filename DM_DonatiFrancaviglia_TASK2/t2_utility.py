import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from t2_constants import *


def elbow_rule(array, n_init=N_INIT, max_iter=MAX_ITER, max_k=MAX_K, figsize=(15,10)):
    sse_list = []

    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter)
        kmeans.fit(array)
        sse_list.append(kmeans.inertia_)

    plt.figure(figsize=figsize)
    plt.plot(range(2, max_k+1), sse_list)
    plt.ylabel('SSE')
    plt.xlabel('K')
    plt.show()


def store_clustering_scores(array, labels, algorithm_name, scores_df):
    SSE = 0
    grouped = pd.DataFrame(array).groupby(pd.Series(labels))

    # computation of SSE
    for _, group in grouped:
        center = group.mean(axis=0)
        SSE += np.sum((group - center)**2).sum()
    scores_df.loc[algorithm_name, 'SSE'] = SSE
    
    # computation of silhouette score
    try:
        scores_df.loc[algorithm_name, 'silhouette'] = silhouette_score(array, labels)
    except ValueError:
        scores_df.loc[algorithm_name, 'silhouette'] = -2


def reoder_labels(labels, centers):
    # Rename clusters labels by cluster size
    renamed_labels = pd.Series(labels.astype(str))
    cluster_sizes = renamed_labels.value_counts()

    for i in renamed_labels.index:
        renamed_labels[i] = cluster_sizes.index.get_loc(renamed_labels[i])

    # Re-order the centers
    centers_copy = pd.DataFrame(centers)  # in case a np array was passed
    renamed_centers = np.zeros_like(centers)
    for i, center in centers_copy.iterrows():
        renamed_centers[cluster_sizes.index.get_loc(str(i))] = center

    return renamed_labels, renamed_centers


def plot_clusters(labels, centers, user_df, figsize):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_xticks([])
    sns.histplot(labels.astype(int).sort_values(), bins=labels.nunique(), ax=ax1)
    items_per_cluster = labels.value_counts()

    for i, center in enumerate(centers):
        if items_per_cluster.loc[i] > np.sqrt(items_per_cluster.nlargest(1).values):
            ax2.plot(center, range(len(user_df.columns)), marker='o', label='Cluster %s' % i)
    
    ax2.set_yticks(range(len(user_df.columns)))
    ax2.set_yticklabels(user_df.columns)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    plt.legend()
    plt.tight_layout()
    plt.show()


def kmeans_run(array, df, algorithm_name, scores_df, n_clusters=None, n_init=N_INIT, max_iter=MAX_ITER, figsize=(20,10), centers=None):
    if centers:
        n_clusters = len(centers)
        kmeans = KMeans(init=centers, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    kmeans.fit(array)
    store_clustering_scores(array, kmeans.labels_, algorithm_name, scores_df)
    kmeans_labels, kmeans_centers = reoder_labels(kmeans.labels_, kmeans.cluster_centers_)
    plot_clusters(kmeans_labels, kmeans_centers, df, figsize)
    plt.show()
    return kmeans_labels


def scatter_cluster(max_cluster, feature_1, feature_2, user_df, figsize=(14,12)):
    # mask to remove points in outliers clusters
    mask = (user_df[LAB] < max_cluster)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(user_df[mask][feature_1], user_df[mask][feature_2], c=user_df[mask][LAB])

    plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1, 1))


def similarity_matrix(array, labels, figsize=(26,24), cmap='ocean'):
    distances = np.zeros((array.shape[0], array.shape[0]))
    for i, x in enumerate(array):
        distances[i, :] = np.sqrt(np.sum((array - x)**2, axis=1))
    distances = distances[np.argsort(labels), :]
    distances = distances[:, np.argsort(labels)]

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(np.clip(distances, 0, 5), ax=ax, cmap=cmap)
    plt.show()
    

def hierarchical_clustering(array, link_method, n_clusters, scores_df, pure=0, algorithm_prefix='hierarchical_', dist_metric='euclidean', link_metric='euclidean', figsize=(15,12)):
    data_dist = pdist(array, metric=dist_metric)
    data_link = linkage(data_dist, method=link_method, metric=link_metric)
    plt.figure(figsize=figsize)
    if pure:
        plt.title(f'Pure hierarchical clustering - {link_method} method')
    else:
        plt.title(f'Hierarchical clustering - {link_method} method')

    dendrogram(data_link, color_threshold=10.0, truncate_mode='lastp')

    algorithm_prefix = 'pure_' + algorithm_prefix if pure else algorithm_prefix
    algorithm_name = algorithm_prefix + link_method

    store_clustering_scores(array, cut_tree(data_link, n_clusters=n_clusters)[:,0], algorithm_name=algorithm_name, scores_df=scores_df)
    return data_link


def show_clusters(data_link, array, df, n_clusters, figsize=(15,12)):
    labels = cut_tree(data_link, n_clusters=n_clusters)[:,0]
    centers = pd.DataFrame(array).groupby(cut_tree(data_link, n_clusters=n_clusters)[:,0]).mean()

    # clustering_plots(labels, centers, user_df.columns)
    new_labels, new_centers = reoder_labels(labels, centers)
    plot_clusters(new_labels, new_centers, df, figsize=figsize)
    return new_labels


def scale_df(df, method):
    if method == 1:
        array = StandardScaler().fit_transform(df)
    else:
        array = MinMaxScaler().fit_transform(df)
    return array, pd.DataFrame(data=array, index=df.index, columns=df.columns)


def create_centers(df):
    k = len(df.columns)
    centers = []
    for i in range(k):
        c = [-1/k]*k
        c[i] = 1
        centers.append(c)
    
    return centers


def plot_expansion(user_df):
    u1 = user_df.quantile(0.05)
    u2 = user_df.quantile(0.25)
    u3 = user_df.quantile(0.5)
    u4 = user_df.quantile(0.75)
    u5 = user_df.quantile(0.95)


    tmp = user_df.groupby(LAB)
    x1 = (tmp.quantile(0.05) - u1)/u1
    x2 = (tmp.quantile(0.25) - u2)/u2
    x3 = (tmp.quantile(0.5) - u3)/u3
    x4 = (tmp.quantile(0.75) - u4)/u4
    x5 = (tmp.quantile(0.95) - u5)/u5

    x_df = x1 + np.sign(x2)*abs(x2)**(1/3) + np.sign(x3)*abs(x3)**(1/5) + np.sign(x4)*abs(x4)**(1/3) + x5

    try:
        x_df = x_df.drop(LAB, axis=1)
    except KeyError:
        pass

    half = int(len(x_df.columns)/2)
    df1 = x_df.iloc[:, :half]
    df2 = x_df.iloc[:, half:]

    df1.plot(figsize=(25,12))
    plt.xticks(ticks=range(user_df[LAB].nunique()), rotation=60)

    df2.plot(figsize=(25,12))
    plt.xticks(ticks=range(user_df[LAB].nunique()), rotation=60)

    x_df.T.plot(figsize=(25,12))
    plt.xticks(ticks=range(len(x_df.columns)), labels=x_df.columns, rotation=60)

    return x_df
