import json
from typing import List, Tuple, Any

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.cluster import KMeans


def cluster_vectors_and_return_strings(vectors: List[np.ndarray], filenames: List[str], k: int) -> Tuple[List[List[str]], np.ndarray]:
    """
    Perform k-means clustering on a list of numpy vectors and return the corresponding names for each cluster.

    :param vectors: A list of numpy vectors.
    :param filenames: A list of names corresponding to the vectors.
    :param k: The number of clusters to form.
    :returns: A list of lists where each inner list contains the strings corresponding to a cluster, and the centroids of the clusters.
    """
    # Convert the list of vectors to a numpy array
    x = np.array(vectors)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, algorithm='elkan', init='k-means++')
    kmeans.fit(x)
    # Get the labels (which cluster each vector belongs to)
    labels = kmeans.labels_
    # Initialize a list of lists to store the strings corresponding to each cluster
    clustered_strings = [[] for _ in range(k)]
    # Populate the list of lists with the corresponding strings
    for idx, label in enumerate(labels):
        clustered_strings[label].append(filenames[idx])
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    return clustered_strings, centroids


def hac_dendrogram(cluster_centroids: np.ndarray, clustered_strings: List[List[Any]]):
    """
    Perform Hierarchical Agglomerative Clustering (HAC) on the cluster centroids and return the dendrogram as a nested dictionary.

    :param cluster_centroids: The centroids of the clusters.
    :param clustered_strings: A list of names corresponding to the cluster centroids.
    :return: A nested dictionary representing the dendrogram. Each node is a dictionary with a 'children' field.
    """
    # Perform HAC
    dendrogram_matrix = linkage(cluster_centroids, method='average')
    # Convert the linkage matrix to a tree structure
    root_node, n_leaves = to_tree(dendrogram_matrix, rd=True)
    # Helper function to recursively build the dendrogram
    def build_dendrogram(node):
        # If it's a leaf node, return the corresponding strings
        if node.is_leaf():
            return {'children': clustered_strings[node.id], 'centroid': list(cluster_centroids[node.id])}
        # Otherwise, recursively build the 'children' field
        return {
            'children': [
                build_dendrogram(node.get_left()),
                build_dendrogram(node.get_right())
            ]
        }
    # Build the dendrogram from the root node
    return  build_dendrogram(root_node)


def save_as_json(data, filename: str):
    """
    Saves some index_metadata to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
