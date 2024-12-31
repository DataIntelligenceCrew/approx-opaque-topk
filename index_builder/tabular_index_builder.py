from typing import List, Tuple, Dict

import time
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
import json
import argparse
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

from builder import save_as_json


"""
Constructs an index by loading a CSV file as a DataFrame, then performing k-means clustering over the rows. 
The CSV file should have numeric, normalized columns. 

USAGE: python tabular_index_builder.py --input_file <path_to_input_file> -k <number_of_clusters>
    --dendrogram_file <path_to_dendrogram_output_file> --flattened_file <path_to_flattened_output_file>
    --subsample_size <number_of_rows_to_subsample> --id_column <name_of_id_column> --pred_column <name_of_prediction_column>
"""


def dataframe_to_matrix_excluding_columns(df: pd.DataFrame, exclude_columns: List) -> np.ndarray:
    """
    Converts a DataFrame into a matrix (numpy array), excluding specified columns.
    Assumes all columns to include are numeric and no missing values.

    :param df: Input DataFrame.
    :param exclude_columns: List of columns to exclude from the DataFrame before conversion.
    :return: A numpy array of the data excluding the specified columns.
    """
    return df.drop(columns=exclude_columns).to_numpy()


def kmeans_clustering(df: pd.DataFrame, n_clusters: int, subsample_size: int, exclude_columns: List[str]) -> Tuple[List[pd.DataFrame], np.ndarray]:
    """
    Performs constrained k-means clustering on a subsample of rows from the DataFrame and assigns all rows
    in the original DataFrame to clusters. Returns a list of DataFrames representing the clusters,
    as well as the centroids of the clusters.

    :param df: The original DataFrame (with headers).
    :param n_clusters: The number of clusters to form.
    :param subsample_size: The number of rows to use for subsampling and performing k-means clustering.
    :param exclude_columns: List of columns to exclude from the clustering process.
    :return: A tuple (list of DataFrames for each cluster, array of centroids).
    """
    # Step 1: Subsample from the original DataFrame
    subsample_df: pd.DataFrame = df.sample(n=subsample_size, random_state=42)
    data_matrix_subsample: np.ndarray = dataframe_to_matrix_excluding_columns(subsample_df, exclude_columns)

    # Step 2: Perform k-means clustering on the subsample
    kmeans = KMeans(n_clusters=n_clusters, verbose=1)
    kmeans.fit(data_matrix_subsample)
    centroids: np.ndarray = kmeans.cluster_centers_

    # Step 3: Assign all rows in the original DataFrame to the nearest cluster centroid
    data_matrix_full: np.ndarray = dataframe_to_matrix_excluding_columns(df, exclude_columns)
    labels_full, _ = pairwise_distances_argmin_min(data_matrix_full, centroids)

    # Step 4: Create a list to hold DataFrames for each cluster
    clusters: List[pd.DataFrame] = []

    # Assign the rows to their respective clusters in the full DataFrame
    for cluster in range(n_clusters):
        cluster_df: pd.DataFrame = df[labels_full == cluster]
        clusters.append(cluster_df)

    return clusters, centroids


def agglomerative_clustering_and_build_tree(centroids: np.ndarray, clusters: List[pd.DataFrame], id_column: str) -> Dict:
    """
    Perform hierarchical agglomerative clustering on the centroids and build a human-readable tree structure.
    Leaf nodes store the values from the ID column.

    :param centroids: The cluster centroids from k-means.
    :param clusters: The list of DataFrames representing each cluster.
    :param id_column: The name of the ID column to store at leaf nodes.
    :return: A dictionary representing the tree structure.
    """
    # Perform HAC
    Z: np.ndarray = linkage(centroids, method='average')
    root_node, _ = to_tree(Z, rd=True)

    # Build the tree dictionary; re-implements the logic since we have leaf as dataframes and not lists
    def build_tree_dict(node) -> dict:
        if node.is_leaf():
            # Leaf node: return the ID column as a list
            cluster_df = clusters[node.id]
            ids = cluster_df[id_column].tolist()
            ids = [str(id) for id in ids]
            return {
                'children': ids
            }
        else:
            # Intermediate node: recursively build its children
            return {
                'children': [
                    build_tree_dict(node.get_left()),
                    build_tree_dict(node.get_right())
                ]
            }

    # Step 4: Build and return the tree structure starting from the root
    return build_tree_dict(root_node)


def build_single_node_tree(df: pd.DataFrame, id_column: str) -> Dict:
    """
    Build a single-node tree with the list of all IDs from the DataFrame.

    :param df: The original DataFrame.
    :param id_column: The name of the ID column to store in the single-node tree.
    :return: A dictionary representing the single-node tree with all IDs.
    """
    ids = df[id_column].tolist()
    return {
        'children': ids
    }


def main(file_path: str, n_clusters: int, tree_output_path: str, single_node_tree_output_path: str,
         subsample_size: int, id_column: str, pred_column: str):
    """
    Main function to run the entire pipeline:
    1. Load CSV
    2. Perform constrained k-means clustering
    3. Build dendrogram tree from k-means clusters, storing the ID column at the leaf nodes
    4. Build single-node tree storing the full list of IDs from the original DataFrame
    5. Save both trees to specified file paths

    :param file_path: Path to the input CSV file.
    :param n_clusters: Number of clusters for k-means.
    :param tree_output_path: Path to save the dendrogram tree structure.
    :param single_node_tree_output_path: Path to save the single-node tree structure.
    :param subsample_size: The number of rows to use for subsampling in k-means.
    :param id_column: The name of the ID column to exclude from clustering and store in the tree.
    """
    # Load CSV as df
    df = pd.read_csv(file_path)
    print("Loaded CSV file")

    # Perform k-means, but exclude the ID column and the prediction column
    exclude_columns = [id_column, pred_column]
    clusters, centroids = kmeans_clustering(df, n_clusters, subsample_size, exclude_columns)

    # Build dendrogram from the k-means clusters, storing only IDs
    dendrogram_tree = agglomerative_clustering_and_build_tree(centroids, clusters, id_column)

    # Build the flat tree structure storing all IDs
    single_node_tree = build_single_node_tree(df, id_column)

    # Step 5: Save both trees to the specified file paths
    save_as_json(dendrogram_tree, tree_output_path)
    save_as_json(single_node_tree, single_node_tree_output_path)


if __name__ == "__main__":
    start_time = time.time()

    # Set up argparse
    parser = argparse.ArgumentParser(description="Run k-means clustering, generate trees, and save output files.")

    # Add arguments for command-line inputs
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("-k", type=int, required=True, help="Number of clusters for k-means.")
    parser.add_argument("--dendrogram_file", type=str, required=True,
                        help="File path to save the dendrogram tree structure.")
    parser.add_argument("--flattened_file", type=str, required=True,
                        help="File path to save the single-node tree structure.")
    parser.add_argument("--subsample_size", type=int, required=True,
                        help="Number of rows to subsample for k-means clustering.")
    parser.add_argument("--id_column", type=str, required=True,
                        help="The name of the ID column to exclude from clustering and store in the tree.")
    parser.add_argument('--pred_column', type=str, required=True)

    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(
        file_path=args.input_file,
        n_clusters=args.k,
        tree_output_path=args.dendrogram_file,
        single_node_tree_output_path=args.flattened_file,
        subsample_size=args.subsample_size,
        id_column=args.id_column,
        pred_column=args.pred_column
    )

    end_time = time.time()

    print("TOTAL TIME:", end_time - start_time)
