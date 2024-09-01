import os
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
from typing import List

def get_image_vectors_from_directory(directory_name, model, processor):
    images_vectors = []
    for f in os.listdir(directory_name):
        if f.endswith('.png'):
            path = os.path.join(directory_name, f)
            image = Image.open(path).convert("RGB")
            image_tensor = processor(image).unsqueeze(0)
            with torch.no_grad():
                vector = model.forward_features(image_tensor).cuda().numpy().flatten()
                images_vectors.append((f, vector))

def hierarchical_clustering(vectors: List[np.ndarray], n_clusters: int):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(vectors)
    children = model.children_

    def build_tree(node_id, current_depth):
        if current_depth == depth:
            return {'elements': [node_id]}
        left_child, right_child = children[node_id]
        left_tree = build_tree(left_child, current_depth + 1)
        right_tree = build_tree(right_child, current_depth + 1)
        return {'children': [left_tree, right_tree]}

    return build_tree(len(vectors) - 2, 0)

def save_as_json(data, filename):
    """Save the dictionary as a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main(directory, output_json, depth=3):
    # Load the BLIP model and preprocessors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_preprocess(name="blip-base", model_type="feature_extractor", is_eval=True, device=device)

    # Scan the directory for images and process them
    image_paths = get_png_images(directory)
    vectors = []
    filenames = []
    for image_path in image_paths:
        vector = get_image_vector(image_path, model, processor)
        vectors.append(vector)
        filenames.append(os.path.basename(image_path))

    # Perform hierarchical clustering and truncate at the specified depth
    tree = hierarchical_clustering(np.array(vectors), depth)

    # Replace indices with filenames in the tree structure
    def replace_indices_with_filenames(node):
        if 'elements' in node:
            node['elements'] = [filenames[idx] for idx in node['elements']]
        if 'children' in node:
            for child in node['children']:
                replace_indices_with_filenames(child)

    replace_indices_with_filenames(tree)

    # Save the tree as a JSON file
    save_as_json(tree, output_json)

if __name__ == "__main__":
    directory = "path/to/your/images"
    output_json = "output.json"
    depth = 3  # You can modify the depth as needed
    main(directory, output_json, depth)
