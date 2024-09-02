import json
import sys
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage


def load_json(json_file):
    """Load the JSON data from the file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def build_linkage_matrix(node, node_list, current_id):
    """Recursively build a linkage matrix and node list from the tree structure."""
    if 'children' in node and node['children']:
        left_id = current_id
        current_id += 1
        right_id = current_id
        current_id += 1

        # Recursively process children
        left_id = build_linkage_matrix(node['children'][0], node_list, left_id)
        right_id = build_linkage_matrix(node['children'][1], node_list, right_id)

        # Internal node
        new_id = len(node_list)
        Z.append([left_id, right_id, 1.0, len(node_list)])
        node_list.append({'id': new_id, 'elements': node.get('elements', [])})
        return new_id
    else:
        # Leaf node
        node_id = len(node_list)
        node_list.append({'id': node_id, 'elements': node.get('elements', [])})
        return node_id


def plot_leaf_images(node_list, ax, leaf_label_func):
    """Plot images under the leaf nodes."""
    ivl = ax.get_xticklabels()
    for label in ivl:
        label_index = int(label.get_text())
        elements = node_list[label_index]['elements']
        sample_images = random.sample(elements, min(len(elements), sample_size))
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(directory_path, img_name)
            if os.path.isfile(img_path):
                try:
                    image = Image.open(img_path)
                    image.thumbnail((30, 30), Image.ANTIALIAS)
                    imagebox = OffsetImage(image, zoom=0.5)
                    ab = AnnotationBbox(imagebox, (label.get_position()[0], -0.5 - i * 0.1), frameon=False)
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")
            else:
                print(f"Image file {img_name} does not exist.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 visualize_index.py /directory/path index_file.json sample_size output_filename.png")
        sys.exit(1)

    directory_path = sys.argv[1]
    json_file = sys.argv[2]
    sample_size = int(sys.argv[3])
    output_filename = sys.argv[4]

    # Load the JSON tree
    try:
        tree = load_json(json_file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    # Initialize the linkage matrix and node list
    Z = []
    node_list = []

    # Build the linkage matrix and node list
    build_linkage_matrix(tree, node_list, current_id=0)

    # Plot dendrogram using scipy
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    dendrogram(Z, ax=ax, labels=[str(node['id']) for node in node_list], leaf_rotation=90, leaf_font_size=10)

    # Plot images under leaf nodes
    plot_leaf_images(node_list, ax, leaf_label_func=None)

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Dendrogram saved to {output_filename}")
