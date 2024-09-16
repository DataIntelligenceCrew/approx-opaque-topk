import json
import sys

from bandit.bandit import approx_top_k_bandit
from bandit.sampler import get_sampler_from_params
from bandit.scorers import get_scorer_from_params


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
    gt_config = all_configs[0]
    gt_result = approx_top_k_bandit(
        index_filename=gt_config['index'],
        k=gt_config['k'],
        scoring_fn=get_scorer_from_params(gt_config['scoring_params']),
        scoring_params=gt_config['scoring_params'],
        sampling_fn=get_sampler_from_params(gt_config['sampling_params']),
        sampling_params=gt_config['sampling_params'],
        algorithm=gt_config['algo_params']['type'],
        algo_params=gt_config['algo_params'],
        budget=gt_config['budget']
    )
    print("Ran GT")
    # Sort sample IDs in descending order
    scores_and_keys = [(iter_result['score'], iter_result['sample_id']) for iter_result in gt_result]
    scores_and_keys.sort(reverse=True)
    # Compute the gt statistics
    gt_id_to_ranking = {}
    for idx, (score, key) in enumerate(scores_and_keys):
        gt_id_to_ranking[key] = idx+1
    # Keep aside the gt solution
    last_iter = gt_result[-1]
    gt_solution = last_iter['pq_elements']
    # Add gt aggregate statistics to stats
    config_stats[gt_config['name']] = {
        "STK": [sum(iter_result['pq_scores']) for iter_result in gt_result],
        "KLS": [iter_result['pq_scores'][-1] for iter_result in gt_result],
        "time": [iter_result['time'] for iter_result in gt_result],
        "precision": [len([elem for elem in iter_result['pq_elements'] if elem in gt_solution]) / gt_config['k']
                      for iter_result in gt_result],
        "recall": [len([elem for elem in gt_solution if elem in iter_result['pq_elements']]) / gt_config['k']
                   for iter_result in gt_result],
        "avg_rank": [sum([gt_id_to_ranking[elem] for elem in iter_result['pq_elements']])
                     / len(iter_result['pq_elements']) for iter_result in gt_result],
        "kth_rank": [gt_id_to_ranking[iter_result['pq_elements'][-1]] for iter_result in gt_result]
    }

    for experiment_config in all_configs[1:]:  # Run all remaining experiments
        expr_reps = experiment_config['reps']
        expr_name = experiment_config['name']
        expr_budget = experiment_config['budget']
        for rep in range(expr_reps):  # Repeat for specified number of runs
            run_result = approx_top_k_bandit(
                index_filename=experiment_config['index'],
                k=experiment_config['k'],
                scoring_fn=get_scorer_from_params(experiment_config['scoring_params']),
                scoring_params=experiment_config['scoring_params'],
                sampling_fn=get_sampler_from_params(experiment_config['sampling_params']),
                sampling_params=experiment_config['sampling_params'],
                algorithm=experiment_config['algo_params']['type'],
                algo_params=experiment_config['algo_params'],
                budget=expr_budget
            )
            # Compute statistics for this run
            iter_stks = [sum(iter_result['pq_scores']) for iter_result in run_result]
            iter_klss = [iter_result['pq_scores'][-1] for iter_result in run_result]
            iter_times = [iter_result['time'] for iter_result in run_result]
            iter_precisions = [len([elem for elem in iter_result['pq_elements']]) / gt_config['k']
                               for iter_result in run_result]
            iter_recalls = [len([elem for elem in gt_solution if elem in iter_result['pq_elements']])
                            / gt_config['k'] for iter_result in run_result]
            iter_avg_ranks = [sum([gt_id_to_ranking[elem] for elem in iter_result['pq_elements']])
                              / len(iter_result['pq_elements']) for iter_result in gt_result]
            iter_kth_ranks = [gt_id_to_ranking[iter_result['pq_elements'][-1]] for iter_result in run_result]
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
                prior_stats = config_stats[expr_name]
                updated_stats = {
                    "STK": [prior_stats["STK"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "KLS": [prior_stats["KLS"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "time": [prior_stats["time"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "precision": [prior_stats["precision"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "recall": [prior_stats["recall"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "avg_rank": [prior_stats["avg_rank"][idx] + iter_stks[idx] for idx in range(expr_budget)],
                    "kth_rank": [prior_stats["kth_rank"][idx] + iter_stks[idx] for idx in range(expr_budget)]
                }
                config_stats[expr_name] = updated_stats
            print("Completed", expr_name, "rep", rep)
        # Average out all metrics
        prior_stats = config_stats[expr_name]
        averaged_stats = {
            "STK": [prior_stats["STK"][idx] / expr_reps for idx in range(expr_budget)],
            "KLS": [prior_stats["KLS"][idx] / expr_reps for idx in range(expr_budget)],
            "time": [prior_stats["time"][idx] / expr_reps for idx in range(expr_budget)],
            "precision": [prior_stats["precision"][idx] / expr_reps for idx in range(expr_budget)],
            "recall": [prior_stats["recall"][idx] / expr_reps for idx in range(expr_budget)],
            "avg_rank": [prior_stats["avg_rank"][idx] / expr_reps for idx in range(expr_budget)],
            "kth_rank": [prior_stats["kth_rank"][idx] / expr_reps for idx in range(expr_budget)],
        }
        config_stats[expr_name] = averaged_stats

    # Save results
    with open(output_file, 'w') as file:
        json.dump(config_stats, file, indent=2)
