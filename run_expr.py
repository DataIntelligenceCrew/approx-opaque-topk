import json
import sys

import numpy as np

from bandit.bandit import approx_top_k_bandit
from bandit.sampler import get_sampler_from_params
from bandit.scorers import get_scorer_from_params

gt_config_name = "Scan"

"""
Runs an experiment with a config file as defined in expr_json_spec.
For each config in the experiment, computes the following statistics, as mean, per iteration. 
 - STK
 - KLS
 - Time
 - Precision@K
 - Recall@K
 - Average ranking
 - Kth ranking

In order to do so, we assume that the first config is the ground truth (exhaustive search), ran only once. 

USAGE: python3 bandit_expr.py <config_file> <output_file>
"""
if __name__ == '__main__':
    config_stats = {}

    # Parse arguments
    config_filename = sys.argv[1]
    output_file = sys.argv[2]

    # Import config
    with open(config_filename, 'r') as file:
        all_configs = json.load(file)

    # Run first experiment as ground truth
    gt_config = all_configs[gt_config_name]
    gt_k = gt_config['k']
    gt_result = approx_top_k_bandit(
        index_filename=gt_config['index'],
        k=gt_k,
        scoring_fn=get_scorer_from_params(gt_config['scoring_params']),
        scoring_params=gt_config['scoring_params'],
        sampling_fn=get_sampler_from_params(gt_config['sampling_params']),
        sampling_params=gt_config['sampling_params'],
        algorithm=gt_config['algo_params']['type'],
        algo_params=gt_config['algo_params'],
        budget=gt_config['budget'],
        sample_method=gt_config['sample_method']
    )
    print("Ran GT")
    # Sort sample IDs in descending order
    scores_and_keys = [(iter_result['score'], iter_result['sample_id']) for iter_result in gt_result]
    scores_and_keys.sort(reverse=True)
    gt_max_rank = len(scores_and_keys)+1
    # Compute the gt statistics
    gt_id_to_ranking = {}
    for idx, (score, key) in enumerate(scores_and_keys):
        gt_id_to_ranking[key] = idx+1
    # Keep aside the gt solution
    gt_solution = set(gt_result[-1]['pq_elements'])
    # Add gt aggregate statistics to stats
    config_stats[gt_config_name] = {
        "STK": np.array([iter_result['stk'] for iter_result in gt_result]),
        "KLS": np.array([iter_result['kls'] for iter_result in gt_result]),
        "time": np.array([iter_result['time'] for iter_result in gt_result]),
        "precision": np.array([len([elem for elem in iter_result['pq_elements'] if elem in gt_solution])
                      for iter_result in gt_result]) / gt_k,
        "recall": np.array([len([elem for elem in gt_solution if elem in iter_result['pq_elements']])
                   for iter_result in gt_result]) / gt_k,
        "avg_rank": np.array([sum([gt_id_to_ranking[iter_result['pq_elements'][k_]]
                                   if k_ < len(iter_result['pq_elements']) else gt_max_rank for k_ in range(gt_k)])
                              for iter_result in gt_result]) / gt_k,
        "kth_rank": np.array([float(gt_id_to_ranking[iter_result['pq_elements'][gt_k-1]])
                              if gt_k <= len(iter_result['pq_elements']) else gt_max_rank for iter_result in gt_result])
    }

    for expr_name, expr_config in all_configs.items():  # Run all remaining experiments
        if expr_name == gt_config_name:
            continue
        expr_reps = expr_config['reps']
        expr_budget = expr_config['budget']
        for rep in range(expr_reps):  # Repeat for specified number of runs
            run_result = approx_top_k_bandit(
                index_filename=expr_config['index'],
                k=expr_config['k'],
                scoring_fn=get_scorer_from_params(expr_config['scoring_params']),
                scoring_params=expr_config['scoring_params'],
                sampling_fn=get_sampler_from_params(expr_config['sampling_params']),
                sampling_params=expr_config['sampling_params'],
                algorithm=expr_config['algo_params']['type'],
                algo_params=expr_config['algo_params'],
                budget=expr_budget,
                sample_method=expr_config['sample_method']
            )
            # Compute statistics for this run
            iter_stks = np.array([iter_result['stk'] for iter_result in run_result])
            iter_klss = np.array([iter_result['kls'] for iter_result in run_result])
            iter_times = np.array([iter_result['time'] for iter_result in run_result])
            iter_precisions = np.array([len([elem for elem in iter_result['pq_elements'] if elem in gt_solution])
                               for iter_result in run_result]) / gt_k
            iter_recalls = np.array([len([elem for elem in gt_solution if elem in iter_result['pq_elements']])
                                     for iter_result in run_result]) / gt_k
            iter_avg_ranks = np.array([sum([gt_id_to_ranking[iter_result['pq_elements'][k_]]
                                   if k_ < len(iter_result['pq_elements']) else gt_max_rank for k_ in range(gt_k)])
                              for iter_result in run_result]) / gt_k
            iter_kth_ranks = np.array([float(gt_id_to_ranking[iter_result['pq_elements'][gt_k-1]])
                              if gt_k <= len(iter_result['pq_elements']) else gt_max_rank for iter_result in run_result])
            # Create dictionary entry if not exists
            if rep == 0:
                config_stats[expr_name] = {
                    "STK": iter_stks,
                    "KLS": iter_klss,
                    "time": iter_times,
                    "precision": iter_precisions,
                    "recall": iter_recalls,
                    "avg_rank": iter_avg_ranks,
                    "kth_rank": iter_kth_ranks
                }
            else:  # Otherwise, sum all metrics
                config_stats[expr_name]["STK"] += iter_stks
                config_stats[expr_name]["KLS"] += iter_klss
                config_stats[expr_name]["time"] += iter_times
                config_stats[expr_name]["precision"] += iter_precisions
                config_stats[expr_name]["recall"] += iter_recalls
                config_stats[expr_name]["avg_rank"] += iter_avg_ranks
                config_stats[expr_name]["kth_rank"] += iter_kth_ranks
            print("Completed", expr_name, "rep", rep)

    # Average out all metrics, then convert numpy arrays to list for serialization
    for expr_name, expr_results in config_stats.items():
        expr_reps = all_configs[expr_name]['reps']
        config_stats[expr_name]["STK"] = list(config_stats[expr_name]["STK"] / float(expr_reps))
        config_stats[expr_name]["KLS"] = list(config_stats[expr_name]["KLS"] / float(expr_reps))
        config_stats[expr_name]["time"] = list(config_stats[expr_name]["time"] / float(expr_reps))
        config_stats[expr_name]["precision"] = list(config_stats[expr_name]["precision"] / float(expr_reps))
        config_stats[expr_name]["recall"] = list(config_stats[expr_name]["recall"] / float(expr_reps))
        config_stats[expr_name]["avg_rank"] = list(config_stats[expr_name]["avg_rank"] / float(expr_reps))
        config_stats[expr_name]["kth_rank"] = list(config_stats[expr_name]["kth_rank"] / float(expr_reps))

    # Save results
    with open(output_file, 'w') as file:
        json.dump(config_stats, file, indent=2)
