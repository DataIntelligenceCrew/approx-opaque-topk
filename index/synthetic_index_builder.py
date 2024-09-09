import os
import json
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from typing import List, Tuple
from scipy.cluster.hierarchy import linkage, to_tree
import sys
import time
from pixel_index_builder import *

def generate_random_distributions(k_: int, n: int):
    """
    :param k_: The number of distributions to generate.
    :param n: The number of samples to generate for each distribution.
    :return: A list of samples and the means of the distributions.
    """
    means = np.random.uniform(0.0, 20.0, k)
    stdevs = np.random.uniform(1.0, 10.0, k)
    samples = []
    for _ in range(k_):
        samples_ = []
        for i in range(n):
            samples_.append(np.random.normal(loc=means[_], scale=stdevs[_]))
            samples.append(samples_)
    return samples, means

if __name__ == '__main__':
    """
    Given some directory path, constructs a VOODOO index over all images in that directory. 
    Then, saves the index to a JSON file. 

    USAGE: python3 pixel_index_builder.py <index_file_path> <k> <n>
    """
    start_time = time.time()

    index_file_path = sys.argv[1]
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    # Cluster the vectors and get the corresponding strings
    clusters, means = generate_random_distributions(k, n)

    clustering_time = time.time()

    # Perform HAC on the cluster centroids
    dendrogram = hac_dendrogram(means, clusters)

    hac_time = time.time()

    # Save the dendrogram to a JSON file
    save_as_json(dendrogram, index_file_path)

    end_time = time.time()

    print(f"LOG: Clustering time: {clustering_time - start_time}")
    print(f"LOG: HAC time: {hac_time - clustering_time}")
    print(f"LOG: Saving time: {end_time - hac_time}")
    print(f"LOG: Total time: {end_time - start_time}")