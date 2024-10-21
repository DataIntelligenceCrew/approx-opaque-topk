import argparse
import json
from typing import Dict, List, Tuple, Callable

from bandit.sampler import get_sampler_from_params
from bandit.scorers import get_scorer_from_params

"""
The ground truth (gt) run over a particular query configuration simply scans over all data points while keeping track 
of a running solution. 

USAGE: python3 run_gt.py --config_file <config_file> --output_file <output_file>
"""

def scan_test(index_params: Dict, k: int, scoring_params: Dict, sampling_params: Dict, batch_size: int) -> Dict:
    """
    :param index_params: Has a single 'file' field, which is the location of a flat JSON index file.
    :param k: Cardinality constraint for top-k query.
    :param scoring_params: Parameters for scoring function.
    :param sampling_params: Parameter s for sampling function.
    :param batch_size: Sample up to this number of elements from the cluster at a time.
    :return: The dictionary that will be written as gt log as described above.
    """
    # Load the index as a list of elements in insertion order
    all_elements: List[str] = get_element_list_from_flat_index(index_params)
    n: int = len(all_elements)

    # Initialize bookkeeping
    samples_scores_list: List[Tuple[float, str]] = list()  # A list of (score, sample_id) tuples

    # Obtain the sampling and scoring functions
    scoring_fn: Callable = get_scorer_from_params(scoring_params)
    sampling_fn: Callable = get_sampler_from_params(sampling_params)

    pointer_idx: int = 0
    while pointer_idx < n:
        # Gather batch_size number of sample IDs from list of elements
        next_pointer_idx: int = min(n, pointer_idx+batch_size)
        sample_ids: List[str] = all_elements[pointer_idx:next_pointer_idx]
        pointer_idx = next_pointer_idx

        # Obtain the actual data objects and score them
        inputs: List[object] = sampling_fn(sample_ids, sampling_params)
        scores: List[float] = scoring_fn(inputs, scoring_params)

        # For each score, update the mapping from samples to scores
        for i in range(len(sample_ids)):
            sample_id, sample_score = sample_ids[i], scores[i]
            samples_scores_list.append((sample_score, sample_id))

    # Get gt rankings by sorting the (score, id) list in descending order
    samples_scores_list.sort(reverse=True)

    # Create a mapping from ID to gt ranking
    id_to_rankings = dict()
    for idx in range(n):
        id_to_rankings[samples_scores_list[idx][1]] = idx+1

    # Get gt solution by iterative over the first k elementes of the sorted list
    gt_solution = []
    for idx in range(k):
        gt_solution.append(samples_scores_list[idx][1])

    # Return result
    result = {
        'gt_solution': gt_solution,
        'gt_rankings': id_to_rankings,
        'n': n
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
    parser.add_argument("config_filename", type=str, help="Path to the configuration file")
    parser.add_argument("output_filename", type=str, help="Path to the output file")
    args = parser.parse_args()

    # Load config
    with open(args.config_filename, 'r') as file:
        configs = json.load(file)

    # Run scan
    test_result = scan_test(
        configs['index_params'],
        configs['k'],
        configs['scoring_params'],
        configs['sampling_params'],
        configs['batch_size']
    )

    # Save output
    with open(args.output_filename, 'w') as file:
        json.dump(test_result, file, indent=2)
