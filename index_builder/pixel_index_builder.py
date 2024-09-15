import os
import numpy as np
from PIL import Image
from typing import List, Tuple
import sys
import time

from index_builder.builder import cluster_vectors_and_return_strings, hac_dendrogram, save_as_json


def get_image_vectors_from_directory(directory_name: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Given a directory which holds some images, obtains the image pixel vectors and their filenames.

    :param directory_name: The directory containing the images. The images are assumed to be .png and RGB.
    :returns: A list of image vectors and a list of image filenames.
    """
    vectors, filenames = [], []
    itr = 0
    # Iterate through all images in the directory
    for f in os.listdir(directory_name):
        if f.endswith('.png'):
            if itr % 1000 == 0:
                print(f"LOG: Processing image {itr}")
            # Obtain image filename and processed vector
            path: str = os.path.join(directory_name, f)
            filenames.append(f)  # Save image filename
            image = Image.open(path).convert("RGB")
            vector = np.array(image).flatten() / 255.0
            vectors.append(vector)
            itr += 1
    return vectors, filenames


if __name__ == '__main__':
    """
    Given some directory path, constructs a VOODOO index_builder over all images in that directory. 
    Then, saves the index_builder to a JSON file. 
    
    USAGE: python3 pixel_index_builder.py <image_directory_path> <index_file_path> <k>
    """
    start_time = time.time()

    image_directory_path = sys.argv[1]
    index_file_path = sys.argv[2]
    k = int(sys.argv[3])

    # Get image vectors and filenames
    vectors, filenames = get_image_vectors_from_directory(image_directory_path)

    file_read_time = time.time()

    # Cluster the vectors and get the corresponding strings
    clustered_strings, centroids = cluster_vectors_and_return_strings(vectors, filenames, k)

    clustering_time = time.time()

    # Perform HAC on the cluster centroids
    dendrogram = hac_dendrogram(centroids, clustered_strings)

    hac_time = time.time()

    # Save the dendrogram to a JSON file
    save_as_json(dendrogram, index_file_path)

    end_time = time.time()

    print(f"LOG: File read time: {file_read_time - start_time}")
    print(f"LOG: Clustering time: {clustering_time - file_read_time}")
    print(f"LOG: HAC time: {hac_time - clustering_time}")
    print(f"LOG: Saving time: {end_time - hac_time}")
    print(f"LOG: Total time: {end_time - start_time}")
