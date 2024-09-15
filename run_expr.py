import json
import sys
from typing import Callable, Dict, Union

from bandit.bandit import approx_top_k_bandit
from bandit.index import get_index_from_dict
from bandit.sampler import get_sampler_from_params
from bandit.scorers import get_scorer_from_params


"""
Runs an experiment with a config file as defined in expr_json_spec.
USAGE: python3 bandit_expr.py <config_file> <output_file>
"""
if __name__ == '__main__':
    results = []

    # Parse arguments
    config_filename = sys.argv[1]
    output_file = sys.argv[2]

    # Import config
    with open(config_filename, 'r') as file:
        data = json.load(file)

    # Run experiment
    for experiment_config in data:
        # Parse experiment config
        name: str = experiment_config['name']
        # Scoring function
        scoring_params: Dict = experiment_config['scoring_params']
        scorer_fn: Callable = get_scorer_from_params(scoring_params)
        # Sampling function
        sampling_params: Dict = experiment_config['sampling_param']
        sampling_fn: Callable = get_sampler_from_params(sampling_params)
        # Algorithm
        algo_params: Dict = experiment_config['algo_params']
        algorithm_name: str = algo_params['type']
        # Other parameters
        budget: Union[int, float] = experiment_config['budget']
        reps: int = experiment_config['reps']
        k_: int = experiment_config['k']
        # Index
        index_filename: str = experiment_config['index']


        # Run experiment
        for _ in range(reps):
            result = approx_top_k_bandit(index_filename, k_, scorer_fn, scoring_params, sampling_fn, sampling_params, algorithm_name, algo_params, budget)
            results.append(result)
            print(f"Experiment {name}, rep {_} done.")

    # Save results
    with open(output_file, 'w') as file:
        json.dump(results, file)
