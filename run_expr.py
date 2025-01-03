import argparse
import json
from typing import Set, Dict, List

import numpy as np

from bandit.bandit import approx_top_k_bandit
from bandit.samplers import get_sampler_from_params
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
    gt_solution: Set = set(gt_result['gt_solution'])  # The ground truth solution set
    gt_scores: Dict[str, float] = gt_result['gt_scores']
    gt_rankings: Dict[str, int] = gt_result['gt_rankings']  # A mapping from element ID to ground truth ranking
    gt_max_rank: int = gt_result['n']
    gt_k: int = len(gt_solution)

    # A dictionary containing all configs' results
    config_stats = dict()

    for expr_name, expr_config in all_configs.items():  # Run all remaining experiments
        expr_reps: int = expr_config['reps']
        expr_budget: int = expr_config['budget']

        for rep in range(expr_reps):  # Repeat for specified number of runs
            # Run bandit
            run_result = approx_top_k_bandit(
                index_params=expr_config['index_params'],
                k=expr_config['k'],
                scoring_fn=get_scorer_from_params(expr_config['scoring_params']),
                scoring_params=expr_config['scoring_params'],
                sampling_fn=get_sampler_from_params(expr_config['sampling_params']),
                sampling_params=expr_config['sampling_params'],
                algo_params=expr_config['algo_params'],
                budget=expr_budget,
                sample_method=expr_config['sample_method'],
                batch_size=expr_config['batch_size'],
                gt_rankings=gt_rankings,
                gt_scores=gt_scores,
                gt_solution=gt_solution,
                skip_scoring_fn=expr_config['skip_scoring_fn'],
                fallback_params=expr_config['fallback_params']
            )

            # Compute metrics
            rep_stks: np.array = np.array([x['STK'] for x in run_result['iter_results']])
            rep_klss: np.array = np.array([x['KLS'] for x in run_result['iter_results']])
            rep_times: np.array = np.array([x['time'] for x in run_result['iter_results']])
            rep_precisions: np.array = np.array([x['Precision@K'] for x in run_result['iter_results']])
            rep_recalls: np.array = np.array([x['Recall@K'] for x in run_result['iter_results']])
            rep_avg_ranks: np.array = np.array([x['AvgRank'] for x in run_result['iter_results']])
            rep_worst_ranks: np.array = np.array([x['WorstRank'] for x in run_result['iter_results']])
            rep_overhead_one_time: float = run_result['overhead_one_time']
            rep_overhead_pq: float = run_result['overhead_pq']
            rep_overhead_algo: float = run_result['overhead_algo']
            rep_overhead_scorer: float = run_result['overhead_scorer']
            rep_overhead_other: float = run_result['overhead_other']
            fallback_switch_itr: int = run_result['fallback_switch_itr']


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
                    "overhead_one_time": rep_overhead_one_time,
                    "overhead_algo": rep_overhead_algo,
                    "overhead_pq": rep_overhead_pq,
                    "overhead_scorer": rep_overhead_scorer,
                    "overhead_other": rep_overhead_other,
                    "fallback_switch_itrs": [fallback_switch_itr],
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
                config_stats[expr_name]["overhead_one_time"] += rep_overhead_one_time
                config_stats[expr_name]["overhead_algo"] += rep_overhead_algo
                config_stats[expr_name]["overhead_pq"] += rep_overhead_pq
                config_stats[expr_name]["overhead_scorer"] += rep_overhead_scorer
                config_stats[expr_name]["overhead_other"] += rep_overhead_other
                config_stats[expr_name]["fallback_switch_itrs"].append(fallback_switch_itr)

            print("Completed", expr_name, "rep", rep)

    # Average out all metrics, then convert numpy arrays to list for serialization
    for expr_name, expr_results in config_stats.items():
        expr_reps: int = expr_results['reps']
        config_stats[expr_name]["STK"]: List[float] = list(config_stats[expr_name]["STK"] / float(expr_reps))
        config_stats[expr_name]["KLS"]: List[float] = list(config_stats[expr_name]["KLS"] / float(expr_reps))
        config_stats[expr_name]["time"]: List[float] = list(config_stats[expr_name]["time"] / float(expr_reps))
        config_stats[expr_name]["Precision@K"]: List[float] = list(config_stats[expr_name]["Precision@K"] / float(expr_reps))
        config_stats[expr_name]["Recall@K"]: List[float] = list(config_stats[expr_name]["Recall@K"] / float(expr_reps))
        config_stats[expr_name]["AvgRank"]: List[float] = list(config_stats[expr_name]["AvgRank"] / float(expr_reps))
        config_stats[expr_name]["WorstRank"]: List[float] = list(config_stats[expr_name]["WorstRank"] / float(expr_reps))
        config_stats[expr_name]["overhead_one_time"] = config_stats[expr_name]["overhead_one_time"] / float(expr_reps)
        config_stats[expr_name]["overhead_algo"] = config_stats[expr_name]["overhead_algo"] / float(expr_reps)
        config_stats[expr_name]["overhead_pq"] = config_stats[expr_name]["overhead_pq"] / float(expr_reps)
        config_stats[expr_name]["overhead_scorer"] = config_stats[expr_name]["overhead_scorer"] / float(expr_reps)
        config_stats[expr_name]["overhead_other"] = config_stats[expr_name]["overhead_other"] / float(expr_reps)

    # Save results
    with open(args.output_filename, 'w') as file:
        json.dump(config_stats, file, indent=2)

