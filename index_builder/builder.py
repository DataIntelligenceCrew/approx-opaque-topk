import json
from typing import List, Any, Dict

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree


"""
Implements the HAC procedure which is used across all index builders.
"""


def hac_dendrogram(cluster_centroids: np.ndarray, clustered_strings: List[List[Any]]) -> Dict:
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
    def build_dendrogram(node) -> Dict:
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
