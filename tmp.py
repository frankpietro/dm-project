import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def reoder_labels(labels, centers):
    # Rename clusters labels by cluster size
    renamed_labels = pd.Series(labels.astype(str))
    cluster_sizes = renamed_labels.value_counts()

    def to_new_label(old_label):
        return cluster_sizes.index.get_loc(old_label)

    for i in renamed_labels.index:
        renamed_labels[i] = to_new_label(renamed_labels[i])

    # Re-order the centers
    centers_copy = pd.DataFrame(centers)  # in case a np array was passed
    renamed_centers = np.zeros_like(centers)
    for i, center in centers_copy.iterrows():
        renamed_centers[to_new_label(str(i))] = center

    return renamed_labels, renamed_centers

def plot_clusters(labels, centers, user_df, storage_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(labels.astype('str').sort_values(), bins=10, ax=ax1)
    items_per_cluster = labels.value_counts()

    for i, center in enumerate(centers):
        if items_per_cluster.loc[i] > np.sqrt(items_per_cluster.nlargest(1).values):
            ax2.plot(center, range(len(user_df.columns)), marker='o', label='Cluster %s' % i)
    # ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_yticks(range(len(user_df.columns)))
    ax2.set_yticklabels(user_df.columns)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    plt.legend()
    plt.tight_layout()
    if storage_path:
        plt.savefig(storage_path, bbox_inches='tight')
    plt.show()