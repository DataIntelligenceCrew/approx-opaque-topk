import argparse
import os
import random
import time

#from k_means_constrained import KMeansConstrained
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from builder import hac_dendrogram, save_as_json

"""
Given some directory path, constructs a VOODOO index_builder over all images in that directory. 
Then, saves the index_builder to a JSON file. 

USAGE: python3 pixel_index_builder.py --dendrogram-file <dendrogram_file> --flattened-file <flattened_file> -k <k> 
    --subsample-size <subsample_size> --image-directory <image_directory>
"""


def get_image_vector(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Open an image, preprocess it to the same dimensions, and return its flattened 1-D vector representation.

    :param image_path: The path to the image file.
    :param target_size: Tuple (width, height) to resize the image to.
    :return: Flattened 1-D numpy array representing the image.
    """
    try:
        with Image.open(image_path) as img:
            img: Image = img.resize(target_size).convert('RGB')
            img_array: np.ndarray = np.array(img)
            vector: np.ndarray = img_array.flatten()
        return vector
    except ValueError as e:
        print(f"Error processing {image_path}: {e}")
        return None  # Return None if an image fails to process


def process_single_image(args) -> np.ndarray:
    """
    Helper function for multithreading. Processes a single image.

    :param args: Tuple containing (image_path, target_size).
    :return: Flattened 1-D numpy array representing the image.
    """
    image_path, target_size = args
    return get_image_vector(image_path, target_size)


def subsample_images(directory: str, num_samples: int, target_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Uniformly subsample a specified number of images from the directory,
    preprocess them to the same dimensions, and return their flattened 1-D vector representations.

    :param directory: The directory containing images.
    :param num_samples: Number of images to subsample.
    :param target_size: Tuple (width, height) to resize images to.
    :return: List of flattened 1-D numpy arrays representing the images.
    """
    image_files: List[str] = [os.path.join(directory, fname) for fname in os.listdir(directory)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    num_samples: int = min(num_samples, len(image_files))

    sampled_files: List[str] = random.sample(image_files, num_samples)
    print("Selected a subsample of filenames")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        vectors = [vector for vector in executor.map(process_single_image, [(file, target_size) for file in sampled_files]) if vector is not None]
    return vectors


def perform_kmeans(vectors: list, n_clusters: int) -> np.ndarray:
    """
    Perform constrained k-means clustering over the vectors and return the centroids.

    :param vectors: List of flattened image vectors.
    :param n_clusters: Number of clusters.
    :return: Centroids of the clusters.
    """
    #size_min: int = int(len(vectors) / n_clusters * 0.5)
    #print("size_min:", size_min)
    vectors: np.ndarray = np.array(vectors)
    kmeans = KMeans(
        n_clusters=n_clusters,
        #size_min=size_min,
        verbose=1,
    )
    kmeans.fit(vectors)
    centroids: np.ndarray = kmeans.cluster_centers_
    return centroids


def label_single_image(args) -> Tuple[str, int]:
    """
    Helper function to label a single image with the nearest cluster index.

    :param args: Tuple containing (directory, fname, centroids, target_size).
    :return: Tuple of (filename, cluster index).
    """
    directory, fname, centroids, target_size = args
    file_path: str = os.path.join(directory, fname)
    vector: np.ndarray = get_image_vector(file_path, target_size)

    if vector is None:
        return None, None

    # Compute distances to centroids
    distances: np.ndarray = np.linalg.norm(centroids - vector, axis=1)

    # Assign to nearest centroid
    cluster_idx: int = np.argmin(distances)
    return fname, cluster_idx


def label_images(directory: str, centroids: np.ndarray, target_size: Tuple[int, int]) -> Dict[str, int]:
    """
    Iterate over all images in the directory, label each filename with the cluster index it belongs to.

    :param directory: The directory containing images.
    :param centroids: Centroids of the clusters.
    :param target_size: Tuple (width, height) to resize images to.
    :return: Dictionary mapping filenames to cluster indices.
    """
    # Get list of image filenames
    image_files: List[str] = [fname for fname in os.listdir(directory)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    image_files.sort()

    print("Got list of all image filenames")

    # Use ProcessPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(label_single_image, [(directory, fname, centroids, target_size) for fname in image_files])
        )
    print("Assigned images to clusters")


    # Build dictionary mapping filenames to cluster indices
    filename_to_cluster = {fname: cluster_idx for fname, cluster_idx in results if fname is not None}
    return filename_to_cluster


def create_dendrogram(centroids: np.ndarray, filename_to_cluster: Dict) -> Dict:
    """
    Turn the clusters into a dendrogram based on the cluster centroids.

    :param centroids: Centroids of the clusters.
    :param filename_to_cluster: Dictionary mapping filenames to cluster indices.
    :return: Dendrogram as a nested dictionary.
    """
    # Prepare clustered_strings: list of lists of filenames in each cluster
    n_clusters: int = centroids.shape[0]
    clustered_strings: List[List[str]] = [[] for _ in range(n_clusters)]
    for fname, cluster_idx in filename_to_cluster.items():
        clustered_strings[cluster_idx].append(fname)
    # Use the provided hac_dendrogram function
    dendrogram: Dict = hac_dendrogram(centroids, clustered_strings)
    return dendrogram


if __name__ == '__main__':
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dendrogram-file', type=str, required=True)
    parser.add_argument('--flattened-file', type=str, required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('--subsample-size', type=int, required=True)
    parser.add_argument('--image-directory', type=str, required=True)
    args = parser.parse_args()
    print("Parsed arguments", time.time() - start_time)

    # Set n manually
    n = 320291

    # Subsample images from directory, then apply k-means over it
    subsample = subsample_images(args.image_directory, args.subsample_size, (16, 16))

    print("Obtained subsample vectors", time.time() - start_time)

    centroids = perform_kmeans(subsample, args.k)

    print("Applied kmeans", time.time() - start_time)

    # Label all image in the directory with a cluster idx
    image_labels = label_images(args.image_directory, centroids, (16, 16))

    print("Labeled images", time.time() - start_time)

    # Construct dendrogram over all the images and their clusters
    dendrogram = create_dendrogram(centroids, image_labels)

    print("Constructed dendrogram", time.time(), start_time)

    # Save dendrogram index
    save_as_json(dendrogram, args.dendrogram_file)

    print("Saved dendrogram", time.time() - start_time)

    # Construct a flat cluster
    flattened_cluster = []
    for image_name in image_labels:
        flattened_cluster.append(image_name)

    save_as_json({'children': [x for x in flattened_cluster]}, args.flattened_file)

    end_time = time.time()

    print("TOTAL TIME:", end_time - start_time)
