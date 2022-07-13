import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from col_names import *


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
    scores_df.loc[algorithm_name, 'silhouette'] = silhouette_score(array, labels)


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
    sns.histplot(labels.astype('str').sort_values(), bins=10, ax=ax1)
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


def kmeans_run(array, df, n_clusters, algorithm_name, scores_df, n_init=N_INIT, max_iter=MAX_ITER, figsize=(20,10)):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    kmeans.fit(array)
    store_clustering_scores(array, kmeans.labels_, algorithm_name, scores_df)
    kmeans_labels, kmeans_centers = reoder_labels(kmeans.labels_, kmeans.cluster_centers_)
    plot_clusters(kmeans_labels, kmeans_centers, df, figsize)
    plt.show()
    return kmeans_labels


def scatter_cluster(max_cluster, feature_1, feature_2, labels, user_df, figsize=(14,12)):
    # mask to remove points in outliers clusters
    mask = (labels < max_cluster)
    mask.index = range(1, len(mask)+1)
    labels.index = range(1, len(labels)+1)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(user_df[mask][feature_1], user_df[mask][feature_2], c=labels[mask])

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