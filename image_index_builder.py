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


def get_less_used_gpu(gpus=None, debug=False):
    if not torch.cuda.is_available():
        return None
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(torch.cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(torch.cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {torch.cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = torch.cuda.memory_allocated(i)
        cur_cached_mem[i] = torch.cuda.memory_reserved(i)
        max_allocated_mem[i] = torch.cuda.max_memory_allocated(i)
        max_cached_mem[i] = torch.cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)


def get_image_vectors_from_directory(directory_name: str, debug_print_: bool, batch_size: int = 10000) -> List[Tuple[str, np.ndarray]]:
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
                images_vectors.append((f, vector.cpu().detach().numpy()))
                features = None
                vector = None
                image = None
                image_tensor = None
                path = None
            if iter_ % batch_size == batch_size - 1:
                model = None
                processors = None
                gc.collect()
                torch.cuda.empty_cache()
            iter_ += 1
    return images_vectors


def hierarchical_clustering(vectors: List[np.ndarray], n_clusters: int):
    model: AgglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(vectors)
    children: np.ndarray = model.children_

    def build_tree(node_id: int, current_depth: int):
        if current_depth == n_clusters - 1:
            return {'elements': [node_id]}
        left_child, right_child = children[node_id]
        left_tree = build_tree(left_child, current_depth + 1)
        right_tree = build_tree(right_child, current_depth + 1)
        return {'children': [left_tree, right_tree]}

    tree = build_tree(len(vectors) - 2, 0)
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
    # Usage: python3 image_index_builder.py <directory_path> <output_json_filename> <n_clusters> <DEBUG>
    directory_path = sys.argv[1]
    output_json_filename = sys.argv[2]
    n_clusters = int(sys.argv[3])
    debug_ = bool(sys.argv[4])
    construct_hac_index_images(directory_path, output_json_filename, n_clusters, debug_)
