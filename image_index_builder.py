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


def get_image_vectors_from_directory(directory_name: str, debug_print_: bool, batch_size: int = 25000) -> Tuple[List[np.ndarray], List[str]]:
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
            if iter_ % 500 == 0:
                print(iter_)
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


def bisecting_kmeans_clustering(vectors: List[np.ndarray], names: List[str], max_depth: int, num_samples: int) -> Dict:
    # Subsample the input list
    subsample_indices = random.sample(range(len(vectors)), num_samples)
    subsample_vectors = [vectors[i] for i in subsample_indices]
    subsample_names = [names[i] for i in subsample_indices]

    centroids = []

    def build_tree(vectors: List[np.ndarray], names: List[str], current_depth: int) -> Dict:
        # If the current depth exceeds max_depth, return all elements in the subtree
        if current_depth >= max_depth or len(vectors) <= 1:
            return {'elements': names}

        # Apply bisecting k-means
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(vectors)
        centroids.append(kmeans.cluster_centers_)

        left_indices = [i for i in range(len(vectors)) if kmeans.labels_[i] == 0]
        right_indices = [i for i in range(len(vectors)) if kmeans.labels_[i] == 1]

        left_vectors = [vectors[i] for i in left_indices]
        right_vectors = [vectors[i] for i in right_indices]
        left_names = [names[i] for i in left_indices]
        right_names = [names[i] for i in right_indices]

        left_tree = build_tree(left_vectors, left_names, current_depth + 1)
        right_tree = build_tree(right_vectors, right_names, current_depth + 1)

        return {'children': [left_tree, right_tree]}

    # Build tree for subsample
    tree = build_tree(subsample_vectors, subsample_names, 0)

    # Assign original elements to the leaves of the tree
    def assign_elements_to_leaves(node: Dict, vector: np.ndarray, name: str):
        if 'elements' in node:
            node['elements'].append(name)
        else:
            left_centroid, right_centroid = node['children'][0]['centroid'], node['children'][1]['centroid']
            left_dist = np.linalg.norm(vector - left_centroid)
            right_dist = np.linalg.norm(vector - right_centroid)
            if left_dist < right_dist:
                assign_elements_to_leaves(node['children'][0], vector, name)
            else:
                assign_elements_to_leaves(node['children'][1], vector, name)

    # Initialize elements field in leaf nodes
    def initialize_elements_field(node: Dict):
        if 'elements' in node:
            node['elements'] = []
        else:
            initialize_elements_field(node['children'][0])
            initialize_elements_field(node['children'][1])

    # Initialize elements field for each leaf
    initialize_elements_field(tree)

    # Assign each original element to the appropriate leaf
    for vector, name in zip(vectors, names):
        assign_elements_to_leaves(tree, vector, name)

    return tree


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
