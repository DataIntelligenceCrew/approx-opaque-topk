import sys
from index_builder import *
import time


# USAGE: python hierarchical_kmeans_builder.py <directory_path> <output_json_filename> <depth> <k>
if __name__ == "__main__":
    start_time = time.time()
    directory_path = sys.argv[1]
    output_json_filename = sys.argv[2]
    depth = int(sys.argv[3])
    k = int(sys.argv[4])

    print(f"LOG: Constructing hierarchical k-means index for images in {directory_path} with branching factor {k} and depth {depth}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LOG: Using device {device}")

    # Get image vectors
    print(f"LOG: Starting to load image vectors from {directory_path}")
    vectors, filenames = get_image_vectors_from_directory(directory_path, True)
    vector_load_time = time.time()
    print(f"LOG: Finished loading {len(vectors)} image vectors from {directory_path} in {vector_load_time - start_time} seconds, {vector_load_time - start_time} seconds total")

    # Perform HAC
    print(f"LOG: Starting hierarchical k-means with branching factor {k} and depth {depth}")
    tree = hierarchical_kmeans(vectors, filenames, depth, k)
    clustering_end_time = time.time()
    print(f"LOG: Finished hierarchical k-means in {clustering_end_time - vector_load_time} seconds, {clustering_end_time - start_time} seconds total")

    # Save tree as JSON
    print(f"LOG: Starting to save tree as JSON to {output_json_filename}")
    save_as_json(tree, output_json_filename)
    save_end_time = time.time()
    print(f"LOG: Finished saving tree as JSON in {save_end_time - clustering_end_time} seconds, {save_end_time - start_time} seconds total")
