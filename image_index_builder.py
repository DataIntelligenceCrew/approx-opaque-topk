import os
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
from typing import List, Callable, Tuple, Dict
import sys
import gc
import time


def del_vars(to_del: List):
    for var in to_del:
        del var
    gc.collect()


def free_memory(to_free: List):
    del_vars(to_free)
    gc.collect()
    torch.cuda.empty_cache()


def get_image_vectors_from_directory(directory_name: str, debug_print_: bool, batch_size: int = 25000) -> List[Tuple[str, np.ndarray]]:
    """
    Given a directory which holds some images, runs the images through a model to get their vector representations.

    :param directory_name: The directory containing the images. The images are assumed to be .png and RGB.
    :param processors: The preprocessor which processes the read Image into a tensor.
    :param model: The model to use for feature extraction. Turns a processed image into a vector.
    :returns: A list tuples of the form (filename, vector) where filename is the name of the image and vector is the
              feature vector of the image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(debug_print_, "Using device: " + str(device))
    images_vectors = []


    iter_ = 0
    for f in os.listdir(directory_name):
        if f.endswith('.png'):
            if iter_ % 500 == 0:
                print(iter_)
            if iter_ % batch_size == 0:
                model, processors, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
            path: str = os.path.join(directory_name, f)
            image: Image = Image.open(path).convert("RGB")
            image_tensor: np.ndarray = processors['eval'](image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.extract_features({"image": image_tensor, "text_input": ""}, mode="image")
                vector = features.image_embeds_proj  # Extract low-dimensional feature vector only
                images_vectors.append((f, vector.cpu().detach().numpy().flatten()))
            del_vars([features, vector, image, image_tensor, path])
            if iter_ % batch_size == batch_size - 1:
                free_memory([model, processors])
            iter_ += 1
    return images_vectors


def hierarchical_clustering(vectors: List[np.ndarray], max_depth: int) -> Dict:
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(vectors)
    children = model.children_
    n_samples = len(vectors)

    def build_tree(node_id: int, current_depth: int) -> Dict:
        # If the current depth exceeds max_depth, return all elements of the subtree
        if current_depth >= max_depth:
            return {'elements': get_subtree_elements(node_id)}

        # Leaf node (individual sample)
        if node_id < n_samples:
            return {'elements': [node_id]}

        # Internal node (non-leaf)
        left_child, right_child = children[node_id - n_samples]
        left_tree = build_tree(left_child, current_depth + 1)
        right_tree = build_tree(right_child, current_depth + 1)

        return {'children': [left_tree, right_tree]}

    def get_subtree_elements(node_id: int) -> List[int]:
        """ Recursively get all elements under this node in the dendrogram. """
        if node_id < n_samples:
            return [node_id]
        left_child, right_child = children[node_id - n_samples]
        return get_subtree_elements(left_child) + get_subtree_elements(right_child)

    # Start building the tree from the root node
    tree = build_tree(n_samples + len(children) - 1, 0)
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
