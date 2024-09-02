import json
import sys
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_dendrogram(node, ax, x=0, y=0, width=1, depth=0, max_depth=None):
    """Recursive function to plot the dendrogram."""
    if max_depth is None:
        max_depth = get_max_depth(node)

    num_children = len(node.get('children', []))
    if num_children == 0:  # Leaf node
        ax.text(x, y, f'{len(node["elements"])} elements', va='center', ha='center')
        plot_leaf_images(node, ax, x, y - 0.05, width)
        return x, x

    left_x = x
    for i, child in enumerate(node['children']):
        child_x, child_x_end = plot_dendrogram(child, ax, left_x, y - 1, width / num_children, depth + 1, max_depth)
        ax.plot([x, (child_x + child_x_end) / 2], [y, y - 1], c='k')
        left_x = child_x_end + width / (2 * num_children)

    return x - width / 2, left_x - width / 2


def get_max_depth(node):
    """Recursively calculate the maximum depth of the tree."""
    if 'children' not in node:
        return 0
    return 1 + max(get_max_depth(child) for child in node['children'])


def plot_leaf_images(node, ax, x, y, width):
    """Plot images under the leaf nodes."""
    sample_images = random.sample(node['elements'], min(len(node['elements']), sample_size))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(directory_path, img_name)
        image = mpimg.imread(img_path)
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y - i * 0.15), frameon=False, box_alignment=(0.5, 1))
        ax.add_artist(ab)


if __name__ == "__main__":
    # Parse command-line arguments
    directory_path = sys.argv[1]
    json_file = sys.argv[2]
    sample_size = int(sys.argv[3])

    # Load the JSON tree
    tree = load_json(json_file)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()

    # Plot the dendrogram
    plot_dendrogram(tree, ax, width=10)

    # Show the plot
    plt.show()
