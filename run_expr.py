import argparse
import json

import numpy as np

from bandit.bandit import approx_top_k_bandit
from bandit.sampler import get_sampler_from_params
from bandit.scorers import get_scorer_from_params


"""
Runs an experiment with a config file as defined in expr_json_spec.
This should be run after the gt experiments, as it uses the gt statistics to compute metrics. 

USAGE: python run_expr.py <config_filename> <gt_result_filename> <output_filename>
"""
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str, help="Path to the configuration file")
    parser.add_argument("gt_filename", type=str, help="Path to the ground truth result file")
    parser.add_argument("output_filename", type=str, help="Path to the output file")
    args = parser.parse_args()

    # Import config
    with open(args.config_filename, 'r') as file:
        all_configs = json.load(file)

    # Import gt results
    with open(args.gt_filename, 'r') as file:
        gt_result = json.load(file)
    gt_solution = set(gt_result['gt_solution'])  # The ground truth solution set
    gt_rankings = gt_result['gt_rankings']  # A mapping from element ID to ground truth ranking
    gt_max_rank = gt_result['n']
    gt_k = len(gt_solution)

    # A dictionary containing all configs' results
    config_stats = dict()

    for expr_name, expr_config in all_configs.items():  # Run all remaining experiments
        expr_reps = expr_config['reps']
        expr_budget = expr_config['budget']

        for rep in range(expr_reps):  # Repeat for specified number of runs
            # Run bandit
            run_result = approx_top_k_bandit(
                index_params=expr_config['index_params'],
                k=expr_config['k'],
                scoring_fn=get_scorer_from_params(expr_config['scoring_params']),
                scoring_params=expr_config['scoring_params'],
                sampling_fn=get_sampler_from_params(expr_config['sampling_params']),
                sampling_params=expr_config['sampling_params'],
                algorithm=expr_config['algo_params']['type'],
                algo_params=expr_config['algo_params'],
                budget=expr_budget,
                sample_method=expr_config['sample_method'],
                batch_size=expr_config['batch_size'],
                gt_rankings=gt_rankings,
                gt_solution=gt_solution
            )

            # Compute metrics
            rep_stks = np.array([x['STK'] for x in run_result['iter_results']])
            rep_klss = np.array([x['KLS'] for x in run_result['iter_results']])
            rep_times = np.array([x['time'] for x in run_result['iter_results']])
            rep_precisions = np.array([x['Precision@K'] for x in run_result['iter_results']])
            rep_recalls = np.array([x['Recall@K'] for x in run_result['iter_results']])
            rep_avg_ranks = np.array([x['AvgRank'] for x in run_result['iter_results']])
            rep_worst_ranks = np.array([x['WorstRank'] for x in run_result['iter_results']])

            # Create dictionary entry if not exists
            if rep == 0:
                config_stats[expr_name] = {
                    "STK": rep_stks,
                    "KLS": rep_klss,
                    "time": rep_times,
                    "Precision@K": rep_precisions,
                    "Recall@K": rep_recalls,
                    "AvgRank": rep_avg_ranks,
                    "WorstRank": rep_worst_ranks,
                    "reps": expr_reps
                }
            else:  # Otherwise, sum all metrics
                config_stats[expr_name]["STK"] += rep_stks
                config_stats[expr_name]["KLS"] += rep_klss
                config_stats[expr_name]["time"] += rep_times
                config_stats[expr_name]["Precision@K"] += rep_precisions
                config_stats[expr_name]["Recall@K"] += rep_recalls
                config_stats[expr_name]["AvgRank"] += rep_avg_ranks
                config_stats[expr_name]["WorstRank"] += rep_worst_ranks

            print("Completed", expr_name, "rep", rep)

    # Average out all metrics, then convert numpy arrays to list for serialization
    for expr_name, expr_results in config_stats.items():
        expr_reps = expr_results['reps']
        config_stats[expr_name]["STK"] = list(config_stats[expr_name]["STK"] / float(expr_reps))
        config_stats[expr_name]["KLS"] = list(config_stats[expr_name]["KLS"] / float(expr_reps))
        config_stats[expr_name]["time"] = list(config_stats[expr_name]["time"] / float(expr_reps))
        config_stats[expr_name]["Precision@K"] = list(config_stats[expr_name]["Precision@K"] / float(expr_reps))
        config_stats[expr_name]["Recall@K"] = list(config_stats[expr_name]["Recall@K"] / float(expr_reps))
        config_stats[expr_name]["AvgRank"] = list(config_stats[expr_name]["AvgRank"] / float(expr_reps))
        config_stats[expr_name]["WorstRank"] = list(config_stats[expr_name]["WorstRank"] / float(expr_reps))

    # Save results
    with open(args.output_filename, 'w') as file:
        json.dump(config_stats, file, indent=2)
