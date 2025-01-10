import json
import argparse
from collections import defaultdict, Counter

def read_json(file_path):
    """Read a JSON file and return the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def find_ids_with_elements(dendrogram, gt_solution):
    """
    Find the IDs in the dendrogram that contain each element in gt_solution.

    Args:
        dendrogram (dict): The nested dictionary encoding the tree.
        gt_solution (list): List of strings to find in the tree.

    Returns:
        dict: A dictionary mapping each gt_solution element to its corresponding ID.
    """
    id_map = {}

    def traverse(node, node_id):
        if isinstance(node["children"], list) and all(isinstance(child, dict) for child in node["children"]):
            for i, child in enumerate(node["children"]):
                traverse(child, node_id + [i])
        elif isinstance(node["children"], list) and all(isinstance(child, str) for child in node["children"]):  # Leaf node
            for element in node["children"]:
                if element in gt_solution:
                    id_map[element] = node_id

    traverse(dendrogram, [])
    return id_map

def analyze_solution_ids(gt_solution_ids):
    """
    Analyze the frequency of elements in each ID.

    Args:
        gt_solution_ids (dict): A mapping of gt_solution elements to their IDs.

    Returns:
        dict: A dictionary with IDs as keys and frequency counts of elements as values.
    """
    id_counter = defaultdict(list)

    for element, node_id in gt_solution_ids.items():
        id_counter[tuple(node_id)].append(element)

    frequency_analysis = {}
    for node_id, elements in id_counter.items():
        frequency_analysis[node_id] = len(elements)

    return frequency_analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze GT Solution in a dendrogram.")
    parser.add_argument("--gt-filename", required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--dendrogram-index", required=True, help="Path to the dendrogram JSON file.")
    args = parser.parse_args()

    # Read files
    gt_data = read_json(args.gt_filename)
    dendrogram_data = read_json(args.dendrogram_index)

    # Extract gt_solution
    gt_solution = gt_data.get("gt_solution", [])

    # Find IDs containing gt_solution elements
    gt_solution_ids = find_ids_with_elements(dendrogram_data, gt_solution)

    # Analyze IDs
    analysis = analyze_solution_ids(gt_solution_ids)

    # Sort analysis by frequency
    sorted_analysis = sorted(analysis.items(), key=lambda x: -x[1])

    # Print analysis
    print("Analysis of IDs containing gt_solution elements:")
    for node_id, freq in sorted_analysis:
        print(f"ID {node_id}: {freq}")

if __name__ == "__main__":
    main()
