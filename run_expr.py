import argparse
import json
from typing import Set, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from src.bandit import approx_top_k_bandit
from src.bandit import get_sampler_from_params
from src.bandit import get_scorer_from_params

"""
Runs an experiment with a config file as defined in expr_json_spec.
This should be run after the gt experiments, as it uses the gt statistics to compute metrics. 

USAGE: python run_expr.py <config_filename> <gt_result_filename> <output_filename> [--max_threads N]
"""
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str, help="Path to the configuration file")
    parser.add_argument("gt_filename", type=str, help="Path to the ground truth result file")
    parser.add_argument("output_filename", type=str, help="Path to the output file")
    parser.add_argument("--max_threads", type=int, default=1, help="Maximum number of processes to use")
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

    def run_single_experiment(rep, expr_config):
        """Run a single repetition of the experiment."""
        run_result = approx_top_k_bandit(
            index_params=expr_config['index_params'],
            k=expr_config['k'],
            scoring_fn=get_scorer_from_params(expr_config['scoring_params']),
            scoring_params=expr_config['scoring_params'],
            sampling_fn=get_sampler_from_params(expr_config['sampling_params']),
            sampling_params=expr_config['sampling_params'],
            algo_params=expr_config['algo_params'],
            budget=expr_config['budget'],
            sample_method=expr_config['sample_method'],
            batch_size=expr_config['batch_size'],
            gt_rankings=gt_rankings,
            gt_scores=gt_scores,
            gt_solution=gt_solution,
            skip_scoring_fn=expr_config['skip_scoring_fn'],
            fallback_params=expr_config['fallback_params']
        )

        metrics = {
            "STK": np.array([x['STK'] for x in run_result['iter_results']]),
            "KLS": np.array([x['KLS'] for x in run_result['iter_results']]),
            "time": np.array([x['time'] for x in run_result['iter_results']]),
            "Precision@K": np.array([x['Precision@K'] for x in run_result['iter_results']]),
            "Recall@K": np.array([x['Recall@K'] for x in run_result['iter_results']]),
            "AvgRank": np.array([x['AvgRank'] for x in run_result['iter_results']]),
            "WorstRank": np.array([x['WorstRank'] for x in run_result['iter_results']]),
            "overhead_one_time": run_result['overhead_one_time'],
            "overhead_algo": run_result['overhead_algo'],
            "overhead_pq": run_result['overhead_pq'],
            "overhead_scorer": run_result['overhead_scorer'],
            "overhead_other": run_result['overhead_other'],
            "fallback_switch_itr": run_result['fallback_switch_itr']
        }
        return rep, metrics

    for expr_name, expr_config in all_configs.items():  # Run all remaining experiments
        expr_reps: int = expr_config['reps']

        results = []
        with ProcessPoolExecutor(max_workers=args.max_threads) as executor:
            future_to_rep = {executor.submit(run_single_experiment, rep, expr_config): rep for rep in range(expr_reps)}
            for future in as_completed(future_to_rep):
                rep, metrics = future.result()
                results.append(metrics)
                print(f"Completed {expr_name} rep {rep}")

        # Determine the minimum lengths for all metrics
        min_lengths = {
            "STK": min(len(res["STK"]) for res in results),
            "KLS": min(len(res["KLS"]) for res in results),
            "time": min(len(res["time"]) for res in results),
            "Precision@K": min(len(res["Precision@K"]) for res in results),
            "Recall@K": min(len(res["Recall@K"]) for res in results),
            "AvgRank": min(len(res["AvgRank"]) for res in results),
            "WorstRank": min(len(res["WorstRank"]) for res in results)
        }

        # Aggregate results with truncation to the minimum lengths
        aggregated = {
            "STK": list(np.mean([res["STK"][:min_lengths["STK"]] for res in results], axis=0)),
            "KLS": list(np.mean([res["KLS"][:min_lengths["KLS"]] for res in results], axis=0)),
            "time": list(np.mean([res["time"][:min_lengths["time"]] for res in results], axis=0)),
            "Precision@K": list(np.mean([res["Precision@K"][:min_lengths["Precision@K"]] for res in results], axis=0)),
            "Recall@K": list(np.mean([res["Recall@K"][:min_lengths["Recall@K"]] for res in results], axis=0)),
            "AvgRank": list(np.mean([res["AvgRank"][:min_lengths["AvgRank"]] for res in results], axis=0)),
            "WorstRank": list(np.mean([res["WorstRank"][:min_lengths["WorstRank"]] for res in results], axis=0)),
            "overhead_one_time": np.mean([res["overhead_one_time"] for res in results]),
            "overhead_algo": np.mean([res["overhead_algo"] for res in results]),
            "overhead_pq": np.mean([res["overhead_pq"] for res in results]),
            "overhead_scorer": np.mean([res["overhead_scorer"] for res in results]),
            "overhead_other": np.mean([res["overhead_other"] for res in results]),
            "fallback_switch_itrs": [res["fallback_switch_itr"] for res in results],
            "reps": expr_reps
        }

        config_stats[expr_name] = aggregated

    # Save results
    with open(args.output_filename, 'w') as file:
        json.dump(config_stats, file, indent=2)
