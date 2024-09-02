import os
import json
import numpy as np
from sklearn.cluster import KMeans
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
from typing import List, Callable, Tuple, Dict
import sys
import gc
import time
import random


def del_vars(to_del: List):
    for var in to_del:
        del var
    gc.collect()


def free_memory(to_free: List):
    del_vars(to_free)
    gc.collect()
    torch.cuda.empty_cache()


def get_image_vectors_from_directory(directory_name: str, debug_: bool, batch_size: int = 25000) -> Tuple[List[np.ndarray], List[str]]:
    """
    Given a directory which holds some images, runs the images through a model to get their vector representations.

    :param directory_name: The directory containing the images. The images are assumed to be .png and RGB.
    :param processors: The preprocessor which processes the read Image into a tensor.
    :param model: The model to use for feature extraction. Turns a processed image into a vector.
    :returns: A list image vectors and a list of image filenames.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors, filenames = [], []

    iter_ = 0
    model, processors, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    for f in os.listdir(directory_name):
        if f.endswith('.png'):
            if iter_ % 1000 == 0:
                debug_print(debug_, f"Processing image {iter_}")
            # Obtain image filename and processed vector
            path: str = os.path.join(directory_name, f)
            filenames.append(f)  # Save image filename
            image: torch.Tensor = processors['eval'](Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            # Obtain image vector representation
            with torch.no_grad():
                features = model.extract_features({"image": image}, mode="image")
                vector = features.image_embeds_proj  # Extract low-dimensional feature vector only
                vectors.append(vector.cpu().detach().numpy().flatten())  # Save image vector
            # Free objects cached in GPU to avoid memory issues
            del_vars([features, vector, image, image, path])
            # Periodically empty GPU cache
            if iter_ % batch_size == batch_size:
                free_memory([model, processors])
                model, processors, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
            iter_ += 1
    return vectors, filenames


def bisecting_kmeans(vectors: List[np.ndarray], filenames: List[str], num_clusters: int) -> Dict:
    """
    Perform bisecting k-means clustering on a set of vectors to cluster them into a dendrogram.

    :param vectors: A list of vectors to cluster.
    :param filenames: A list of filenames corresponding to the vectors.
    :param num_clusters: The number of clusters to create.
    :returns: A tree representing the clustering. Each intermediate node has a "children" field containing two child nodes.
              Each leaf node has an "elements" field containing the filenames of the vectors in that cluster.
    """
    def kmeans_inner(vectors_: List[np.ndarray], filenames_: List[str], k: int) -> Dict:
        """
        Perform 2-means clustering on a set of vectors to cluster them into two sub-clusters.
        """
        n = len(vectors_)
        if n <= k:  # Base case: There are less elements than the number of clusters. Each cluster is a singleton.
            return {"children": [{"elements": filename} for filename in filenames_]}
        if len(vectors_) == 1:  # Base case: There is only one cluster left. No need to perform clustering work.
            return {"elements": filenames_}
        # Normal case: Divide the existing cluster into two sub-clusters using k-means.
        kmeans = KMeans(n_clusters=2, init='k-means++', algorithm='elkan')
        cluster_assignments = kmeans.fit_predict(vectors_)
        children = []
        for i in range(2):
            child_vectors = [vectors_[j] for j in range(n) if cluster_assignments[j] == i]
            child_filenames = [filenames_[j] for j in range(n) if cluster_assignments[j] == i]
            children.append(kmeans_inner(child_vectors, child_filenames, k-1))
        return {"children": children}

    return kmeans_inner(vectors, filenames, num_clusters)


def save_as_json(data, filename: str):
    """
    Saves some data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def construct_hac_index_images(directory_path: str, output_json_filename, n_clusters: int, debug_print_true: bool):
    debug_print(debug_print_true, f"Constructing HAC index for images in {directory_path} with {n_clusters} clusters")
    # Load the BLIP model and preprocessors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the images' filenames and vectors
    images_vectors = get_image_vectors_from_directory(directory_path, debug_print_true)
    vectors = [images_vectors[1] for images_vectors in images_vectors]
    filenames = [images_vectors[0] for images_vectors in images_vectors]
    debug_print(debug_print_true, "Images processed and vectors extracted")

    # Perform HAC
    tree = hierarchical_clustering(vectors, n_clusters)
    debug_print(debug_print_true, "HAC performed")

    # Replace indices with filenames in the tree structure
    def replace_indices_with_filenames(node):
        if 'elements' in node:
            node['elements'] = [filenames[idx] for idx in node['elements']]
        if 'children' in node:
            for child in node['children']:
                replace_indices_with_filenames(child)

    replace_indices_with_filenames(tree)
    debug_print(debug_print_true, "Replaced indices with filenames in the tree")

    # Save the tree as a JSON file
    save_as_json(tree, output_json_filename)
    debug_print(debug_print_true, f"Saved the tree to {output_json_filename}")


def debug_print(debug_print_true: bool, message: str):
    if debug_print_true:
        print(message)


if __name__ == "__main__":
    start_time = time.time()
    # Usage: python3 image_index_builder.py <directory_path> <output_json_filename> <n_clusters> <DEBUG>
    directory_path = sys.argv[1]
    output_json_filename = sys.argv[2]
    n_clusters = int(sys.argv[3])
    debug_ = bool(sys.argv[4])
    construct_hac_index_images(directory_path, output_json_filename, n_clusters, debug_)
    end_time = time.time()
    debug_print(debug_, "Time: " + str(end_time - start_time))
