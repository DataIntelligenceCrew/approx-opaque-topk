import os
import json
import numpy as np
from sklearn.cluster import KMeans
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
from typing import List, Tuple, Dict
import gc


def del_vars(to_del: List):
    for var in to_del:
        del var
    gc.collect()


def free_memory(to_free: List):
    del_vars(to_free)
    gc.collect()
    torch.cuda.empty_cache()


def get_image_vectors_from_directory(directory_name: str, batch_size: int = 25000) -> Tuple[List[np.ndarray], List[str]]:
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
                print(f"LOG:Processing image {iter_}")
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


def hierarchical_kmeans(vectors: List[np.ndarray], filenames: List[str], max_depth: int, k: int) -> Dict:
    """
    Perform traditional hierarchical k-means on a set of named vectors. The number of leaves is at most k^max_depth.

    :param vectors: A list of vectors to cluster.
    :param filenames: A list of filenames corresponding to the vectors.
    :param max_depth: The maximum depth of the tree.
    :param k: The number of children each node should have.
    :returns: A tree representing the clustering. Each intermediate node has a "children" field containing two child nodes.
              Each leaf node has an "elements" field containing the filenames of the vectors in that cluster.
    """
    def kmeans_inner(vectors_: List[np.ndarray], filenames_: List[str], current_depth: int) -> Dict:
        """
        Perform 2-means clustering on a set of vectors to cluster them into two sub-clusters.
        """
        n = len(vectors_)
        if n <= k:  # Base case: There are fewer elements than the number of clusters. Each cluster is a singleton.
            return {"children": [{"elements": [filename]} for filename in filenames_]}
        if current_depth >= max_depth:  # Base case: The maximum depth has been reached.
            return {"elements": filenames}
        # Normal case: Divide the existing cluster into two sub-clusters using k-means.
        kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='elkan')
        cluster_assignments = kmeans.fit_predict(vectors_)
        children = []
        for i in range(2):
            child_vectors = [vectors_[j] for j in range(n) if cluster_assignments[j] == i]
            child_filenames = [filenames_[j] for j in range(n) if cluster_assignments[j] == i]
            children.append(kmeans_inner(child_vectors, child_filenames, current_depth + 1))
        return {"children": children}

    return kmeans_inner(vectors, filenames, 0)


def bisecting_kmeans(vectors: List[np.ndarray], filenames: List[str], k: int) -> Dict:
    def bisecting_kmeans_inner(tree_: Dict):
        """
        Performs one iteration of bisecting k-means on a tree, which bisects the leaf with the highest inertia.
        """
        leaf_to_modify = tree_
        # Find the child with the highest inertia to bisect
        while 'children' in leaf_to_modify:
            children_priorities = [child['inertia'] for child in leaf_to_modify['children']]
            leaf_to_modify = leaf_to_modify['children'][children_priorities.index(max(children_priorities))]
        # Perform 2-means clustering on the chosen leaf
        vectors_ = leaf_to_modify['vectors']
        filenames_ = leaf_to_modify['filenames']
        n = len(vectors_)
        if n <= k:  # Base case: There are fewer elements than the number of clusters. Each cluster is a singleton.
            leaf_to_modify['children'] = {"children": [{"elements": [filename]} for filename in filenames_]}
            del leaf_to_modify['vectors']
            del leaf_to_modify['filenames']
        else:
            kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='elkan')
            cluster_assignments = kmeans.fit_predict(vectors_)
            children = []
            for i in range(2):
                child_vectors = [vectors_[j] for j in range(n) if cluster_assignments[j] == i]
                child_filenames = [filenames_[j] for j in range(n) if cluster_assignments[j] == i]
                inertia = np.mean([np.linalg.norm(v - kmeans.cluster_centers_[i]) for v in child_vectors])
                children.append({"vectors": child_vectors, 'inertia': inertia, "filenames": child_filenames})
            leaf_to_modify['children'] = children
            del leaf_to_modify['vectors']
            del leaf_to_modify['filenames']
    # Construct base tree, then perform k iterations of bisecting k-means
    tree = {"vectors": vectors, "filenames": filenames, 'inertia': 0}
    for _ in range(k):
        bisecting_kmeans_inner(tree)
    # Remove vectors from the tree, rename filenames to elements
    def remove_vectors(node):
        if 'vectors' in node:
            del node['vectors']
        if 'filenames' in node:
            node['elements'] = node['filenames']
            del node['filenames']
        if 'children' in node:
            for child in node['children']:
                remove_vectors(child)
    remove_vectors(tree)
    return tree


def save_as_json(data, filename: str):
    """
    Saves some data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

