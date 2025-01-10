import json
from typing import List, Any, Dict

import numpy as np
from scipy.spatial.distance import pdist, squareform

"""
Implements a Balanced Hierarchical Agglomerative Clustering (HAC) procedure.
"""

def balanced_hac(cluster_centroids: np.ndarray, clustered_strings: List[List[Any]]) -> Dict:
    """
    Perform Balanced Hierarchical Agglomerative Clustering (HAC) on the cluster centroids.

    :param cluster_centroids: The centroids of the clusters.
    :param clustered_strings: A list of names corresponding to the cluster centroids.
    :return: A nested dictionary representing the dendrogram. Each node is a dictionary with a 'children' field.
    """
    # Initialize clusters as singletons
    clusters = [
        {'children': clustered_strings[i], 'centroid': cluster_centroids[i], 'size': 1}
        for i in range(len(cluster_centroids))
    ]

    # Compute initial distance matrix
    distance_matrix = squareform(pdist(cluster_centroids, metric='euclidean'))
    np.fill_diagonal(distance_matrix, np.inf)  # Prevent self-merges

    def average_linkage(cluster_a, cluster_b):
        """Compute average linkage distance between two clusters."""
        points_a = np.array(cluster_a['centroid']).reshape(-1, len(cluster_centroids[0]))
        points_b = np.array(cluster_b['centroid']).reshape(-1, len(cluster_centroids[0]))
        return np.mean(pdist(np.vstack((points_a, points_b)), metric='euclidean'))

    while len(clusters) > 1:
        # Identify all valid merges that satisfy the balance constraint
        valid_merges = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Check balance constraint
                size_a, size_b = clusters[i]['size'], clusters[j]['size']
                if abs(size_a - size_b) <= 1:
                    valid_merges.append((i, j, distance_matrix[i, j]))

        if not valid_merges:
            raise ValueError("No valid merges found that satisfy the balance constraint.")

        # Find the best merge based on average linkage
        best_merge = min(valid_merges, key=lambda x: x[2])
        i, j, _ = best_merge

        # Merge the two clusters
        new_cluster = {
            'children': [clusters[i], clusters[j]],
            'centroid': (np.array(clusters[i]['centroid']) * clusters[i]['size'] +
                        np.array(clusters[j]['centroid']) * clusters[j]['size']) /
                        (clusters[i]['size'] + clusters[j]['size']),
            'size': clusters[i]['size'] + clusters[j]['size']
        }

        # Remove old clusters and add new one
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j] + [new_cluster]

        # Update distance matrix
        new_distances = []
        for k in range(len(clusters) - 1):
            new_distances.append(average_linkage(clusters[k], new_cluster))
        distance_matrix = np.vstack((distance_matrix[:-1, :-1], new_distances + [np.inf]))
        distance_matrix = np.hstack((distance_matrix, [[np.inf]] * len(distance_matrix)))
        np.fill_diagonal(distance_matrix, np.inf)

    return clusters[0]



def save_as_json(data, filename: str):
    """
    Saves some index_metadata to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
