from typing import List, Tuple

import numpy as np
import argparse

from builder import hac_dendrogram, save_as_json

def generate_random_distributions(k: int, n: int, mu_min: float, mu_max: float, stdev_min: float, stdev_max: float) -> Tuple[List[List[float]], np.ndarray]:
    """
    :param k: The number of distributions to generate.
    :param n: The number of samples to generate for each distribution.
    :param stdev_max: Maximum standard deviation possible.
    :param stdev_min: Minimum standard deviation possible.
    :param mu_max: Maximum mean possible.
    :param mu_min: Minimum mean possible.
    :return: A list of samples and the means of the distributions.
    """
    # Randomly generate means and stdevs
    means = np.random.uniform(mu_min, mu_max, k)
    stdevs = np.random.uniform(stdev_min, stdev_max, k)
    # Draw samples
    samples = []  # A 2-dimensional list of shape (k, n)
    for k_ in range(k):
        scores = np.random.normal(loc=means[k_], scale=stdevs[k_], size=n)
        scores = [x for x in scores]
        samples.append(scores)
    return samples, means

if __name__ == '__main__':
    """
    Given some directory path, constructs a synthetic VOODOO index. 
    In this setting, a leaf cluster is a synthetic normal distribution, where the mean and standard deviations are
    uniformly randomly sampled from a specified range. 
    The dendrogram is built over the means of the distributions. 
    A separate version of the index, which combines all samples into a single leaf, is built as well. 
    The dendrogram-based index and the flat index are saved into separate JSON files. 
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dendrogram-file', type=str, required=True)
    parser.add_argument('--flattened-file', type=str, required=True)
    parser.add_argument('-k', type=int, required=True, help='Number of leaf clusters.')
    parser.add_argument('-n', type=int, required=True, help='Number of samples per leaf cluster.')
    parser.add_argument('--mu-min', type=float, required=False, default=0.0,
                        help='Minimum range to draw mean (mu) from.')
    parser.add_argument('--mu-max', type=float, required=False, default=10.0,
                        help='Maximum range to draw mean (mu) from.')
    parser.add_argument('--stdev-min', type=float, required=False, default=0.001,
                        help='Minimum range to draw standard deviation from.')
    parser.add_argument('--stdev-max', type=float, required=False, default=5.0,
                        help='Maximum range to draw standard deviation from.')
    args = parser.parse_args()

    # Cluster the vectors and get the corresponding strings
    clusters, means = generate_random_distributions(k=args.k, n=args.n, mu_min=args.mu_min, mu_max=args.mu_max, stdev_min=args.stdev_min, stdev_max=args.stdev_max)

    # Flatten the samples in descending order
    flattened_cluster = [x for xs in clusters for x in xs]
    flattened_cluster = sorted(flattened_cluster, reverse=True)
    flattened_cluster = [str(x) for x in flattened_cluster]

    # Obtain GT rank of the samples
    id_to_ranking = {}
    for ranking, sample in enumerate(flattened_cluster):
        sample_id = str(sample)
        id_to_ranking[sample_id] = ranking+1

    # Modify both the nested and flattened clusters to add ranking to each item
    str_clusters = []
    for cluster in clusters:
        str_cluster = [str(x) for x in cluster]
        str_clusters.append(str_cluster)

    # We need to add an extra zero column to the means since the dendrogram clustering method requires 2 or more dimensions to HAC vectors
    zeros = np.zeros(args.k)
    padded_means = np.column_stack((means, zeros))

    # Perform HAC on the cluster centroids
    dendrogram = hac_dendrogram(padded_means, str_clusters)

    # Save the dendrogram to a JSON file
    save_as_json(dendrogram, args.dendrogram_file)

    # Save the flattened index to a JSON file
    save_as_json({'children': flattened_cluster}, args.flattened_file)
