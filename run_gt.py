import argparse
import json
import time
from typing import Dict, List, Tuple, Callable, Any
from tqdm import tqdm  # Import tqdm for progress bar

from bandit.samplers import get_sampler_from_params
from bandit.scorers import get_scorer_from_params

"""
The ground truth (gt) run over a particular query configuration simply scans over all data points while keeping track 
of a running solution. 

USAGE: python3 run_gt.py --config_file <config_file> --gt_result_file <output_file> --sorted_index_file <sorted_index_file>
"""

def scan_test(index_params: Dict, k: int, scoring_params: Dict, sampling_params: Dict, batch_size: int) -> Dict:
    """
    :param index_params: Has a single 'file' field, which is the location of a flat JSON index file.
    :param k: Cardinality constraint for top-k query.
    :param scoring_params: Parameters for scoring function.
    :param sampling_params: Parameters for sampling function.
    :param batch_size: Sample up to this number of elements from the cluster at a time.
    :return: The dictionary that will be written as gt log as described above.
    """
    index_build_start_time: float = time.time()

    # Load the index as a list of elements in insertion order
    all_elements: List[str] = get_element_list_from_flat_index(index_params)
    n: int = len(all_elements)

    # Initialize bookkeeping
    samples_scores_list: List[Tuple[float, str]] = list()  # A list of (score, sample_id) tuples

    # Obtain the sampling and scoring functions
    scoring_fn: Callable = get_scorer_from_params(scoring_params)
    sampling_fn: Callable = get_sampler_from_params(sampling_params)

    pointer_idx: int = 0

    # Initialize tqdm progress bar
    with tqdm(total=n, desc="Scanning", unit="elements") as pbar:
        while pointer_idx < n:
            # Gather batch_size number of sample IDs from list of elements
            next_pointer_idx: int = min(n, pointer_idx + batch_size)
            sample_ids: List[str] = all_elements[pointer_idx:next_pointer_idx]
            pointer_idx = next_pointer_idx

            # Update the progress bar
            pbar.update(len(sample_ids))

            # Obtain the actual data objects and score them
            inputs: List[object] = sampling_fn(sample_ids, sampling_params)
            scores: List[float] = scoring_fn(inputs, scoring_params)

            # For each score, update the mapping from samples to scores
            for i in range(len(sample_ids)):
                sample_id, sample_score = sample_ids[i], scores[i]
                sample_tuple = (sample_score, sample_id)
                samples_scores_list.append(sample_tuple)

    # Get gt rankings by sorting the (score, id) list in descending order
    samples_scores_list.sort(reverse=True)

    # Log the total time required to build a sorted index that has been spent in CPU, with the exception of time to
    # write to disk that will be required later.
    index_build_end_time: float = time.time()
    index_building_cpu_time: float = index_build_end_time - index_build_start_time

    print("Index built")

    # Create a mapping from ID to gt ranking
    id_to_rankings: Dict[Any, int] = dict()
    for idx in range(n):
        id_to_rankings[samples_scores_list[idx][1]] = idx + 1

    # Create a mapping from ID to gt score
    id_to_scores: Dict[Any, float] = dict()
    for idx in range(n):
        id_to_scores[samples_scores_list[idx][1]] = samples_scores_list[idx][0]

    # Get gt solution by iterating over the first k elements of the sorted list
    gt_solution = []
    for idx in range(k):
        gt_solution.append(samples_scores_list[idx][1])

    # Return result
    result: Dict[str, Any] = {
        'gt_solution': gt_solution,
        'gt_rankings': id_to_rankings,
        'gt_scores': id_to_scores,
        'n': n,
        'sorted_list': samples_scores_list,
        'index_building_cpu_time': index_building_cpu_time
    }
    return result

def get_element_list_from_flat_index(index_params: Dict) -> List[str]:
    """
    Reads and returns a list of all element identifiers from a flat index.
    :param index_params: Has a single 'file' field, which is the location of a flat JSON index file.
    :return: A list of element
    """
    # Open index file
    with open(index_params['file'], 'r') as file_:
        index_dict = json.load(file_)
    return index_dict['children']

if __name__ == '__main__':
    # Parse command-line configs
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file to be created.")
    parser.add_argument("output_file", type=str, help="Path to the output file to be created.")
    parser.add_argument("sorted_index_file", type=str, help="Path to the sorted index file to be created.")
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as file:
        configs = json.load(file)

    # Run scan
    test_result = scan_test(
        configs['index_params'],
        configs['k'],
        configs['scoring_params'],
        configs['sampling_params'],
        configs['batch_size']
    )

    # Save sorted index to sorted_index_file
    sorted_index_write_start_time = time.time()
    with open(args.sorted_index_file, 'w') as file:
        sorted_index = {
            'children': [x[1] for x in test_result['sorted_list']]
        }
        json.dump(sorted_index, file, indent=2)
    sorted_index_write_end_time = time.time()
    sorted_index_write_time = sorted_index_write_end_time - sorted_index_write_start_time
    index_total_time = test_result['index_building_cpu_time'] + sorted_index_write_time

    # Save output to output_file
    with open(args.output_file, 'w') as file:
        gt_result = {
            'gt_solution': test_result['gt_solution'],
            'gt_rankings': test_result['gt_rankings'],
            'gt_scores': test_result['gt_scores'],
            'n': test_result['n'],
            'index_time': index_total_time
        }
        json.dump(gt_result, file, indent=2)
