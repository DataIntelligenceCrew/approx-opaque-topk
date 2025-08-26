import json
import math
import random
import time
from tqdm import tqdm
from typing import List, Callable, Union, Dict, Set, Any, Tuple
from pathlib import Path

import numpy as np

from src.bandit.index import IndexNode, IndexLeaf, get_index_from_dict
from src.bandit.limited_pq import LimitedPQ, LimitedList


def approx_top_k_bandit(index_params: Dict, k: int, scoring_fn: Callable, scoring_params: Dict, sampling_fn: Callable,
                        sampling_params: Dict, algo_params: Dict, budget: Union[int, float], sample_method: str,
                        batch_size: int, gt_rankings: Dict, gt_scores: Dict, gt_solution: Set[str],
                        skip_scoring_fn: bool, fallback_params: Dict) -> Dict:
    """
    Perform a run of approximate top-k bandit algorithm.
    This assumes that the index_builder has already been loaded, but not initialized with bandit-related metadata.

    :param index_params: Location of JSON index file.
    :param k: Cardinality constraint for top-k query.
    :param scoring_fn: A function which takes a sample and returns a score.
    :param scoring_params: A dictionary of parameters to pass to the scoring function.
    :param sampling_fn: A function which takes a string sample identifier and returns a sample.
    :param sampling_params: A dictionary of parameters to pass to the sampling function.
    :param algo_params: A dictionary of parameters to pass to the bandit algorithm.
    :param budget: The budget for the bandit run. This can be either an integer (number of iterations) or a float (time in seconds).
    :param sample_method: The method for sampling from the leaf nodes. Either "scan", "replace", or "noreplace".
    :param batch_size: Sample up to this number of elements from the cluster at a time.
    :param gt_rankings: A dictionary mapping from string IDs to ground truth rankings.
    :param gt_scores: A dictionary mapping from string IDs to ground truth scores.
    :param gt_solution: The optimal ground truth solution containing k highest-scoring elements in the dataset.
    :param skip_scoring_fn: If this is set to True, then instead of calling the scoring_fn on the sampled object, instead
                            retrieves the score from the gt_scores dictionary. Runtime will be changed accordingly.
    :return: A dictionary representing the result of this bandit run. It has the following fields:
             - "iter_results": A list of results for each iteration. Each iteration's result is a dictionary with the
                               following statistics:
                - Always: STK, KLS, time, element, score
                - Only if gt_rankings and gt_solution were provided: Precision@K, Recall@K, AvgRank, WorstRank
             - "solution_set": A list of string IDs for the final solution.
             - "index_time": Time it took for the initial index construction overhead.
             - "iter_time": Average time it took for each iteration.
    """
    total_time_per_category: Dict = {
        "once": 0.0,
        "pq": 0.0,
        "algo": 0.0,
        "scorer": 0.0,
        "other": 0.0
    }
    time_elapsed_on_useful_work = 0.0
    fallback_switch_itr = -1


    timestamp: float = time.time()
    ### INITIAL ONE-TIME OVERHEAD START ###

    # Load in the index
    index: Dict = get_initialized_index(index_params['file'], sample_method, algo_params)
    if sample_method == "reversescan":
        sample_method = "scan"

    # Initialize bookkeeping index_metadata structures
    n: int = index.remaining_size()
    itr: int = 1
    pq: Union[LimitedPQ, LimitedList] = LimitedList(k) if sample_method == 'sortedscan' else LimitedPQ(k)
    iter_results: List[Dict] = []
    next_fallback_check_itr = int(n * fallback_params['initial_threshold']) if fallback_params['enabled'] else n+1
    print(f"fallback iterations: {next_fallback_check_itr}")

    ### INITIAL ONE-TIME OVERHEAD END ###
    one_time_overhead: float = time.time() - timestamp
    total_time_per_category['once'] += one_time_overhead
    time_elapsed_on_useful_work += one_time_overhead


    # Main loop over batches of samples
    with tqdm(total=budget) as pbar:
        while True:

            timestamp: float = time.time()
            ### TERMINATION CONDITION START ###
            # Check termination condition, if so terminate
            if termination_condition_is_met(budget, itr):
                break
            # Check if index is empty, if so terminate
            if index.children is None or len(index.children) == 0:
                break
            ### TERMINATION CONDITION END ###
            iter_termination_check_time = time.time() - timestamp
            total_time_per_category['other'] += iter_termination_check_time
            time_elapsed_on_useful_work += iter_termination_check_time


            timestamp = time.time()
            ### FALLBACK LOGIC START ###
            if algo_params['type'] == 'EpsGreedy' and fallback_params['enabled']:
                if itr + batch_size > next_fallback_check_itr:  # We should check fallback condition this iteration
                    s_k = pq.kth_best_score()

                    ### FALLBACK LEVEL 1: DYNAMIC INDEX REBUILDING

                    epsgreedy_leaf_idx: List[int] = select_leaf_arm_epsgreedy(index, algo_params, float('inf'), s_k)
                    epsgreedy_leaf_arm: IndexLeaf = index.get_grandchild(epsgreedy_leaf_idx)
                    epsgreedy_gain = epsgreedy_leaf_arm.metadata['histogram'].expected_marginal_gain(s_k)
                    greedy_gain: float = index.get_greedy_gain(s_k)

                    # Fallback condition is met
                    if greedy_gain > epsgreedy_gain:
                        print(f"Fallback strategy 1 at itr {itr} as greedy gain {greedy_gain} > epsgreedy_gain {epsgreedy_gain}")
                        index.dynamic_rebuild(s_k)

                    ### FALLBACK LEVEL 2: UNIFORMSAMPLE FALLBACK

                    # Estimate the expected marginal gain of the two competing algorithms
                    average_gain: float = index.get_average_gain(s_k)

                    print(f"epsgreedy_gain: {epsgreedy_gain}, greedy_gain: {greedy_gain}, average_gain: {average_gain}")

                    # Estimate the amount of time spent amortized per iteration for the two competing algorithms
                    #uniform_sample_time_per_iter = (total_time_per_category['pq'] + total_time_per_category['scorer'] + total_time_per_category['other']) / itr
                    #epsgreedy_time_per_iter = uniform_sample_time_per_iter + total_time_per_category['algo'] / itr

                    # Estimate the slope of the tangent line at the current iteration for the two competing algorithms
                    uniform_sample_tangent_slope = average_gain #/ uniform_sample_time_per_iter
                    epsgreedy_tangent_slope = greedy_gain #/ epsgreedy_time_per_iter

                    #print(f"DEBUG: itr {itr}, UniformSample slope = {average_gain} / {uniform_sample_time_per_iter} = {uniform_sample_tangent_slope}")
                    #print(f"DEBUG: itr {itr}, EpsGreedy slope = {greedy_gain} / {epsgreedy_time_per_iter} = {epsgreedy_tangent_slope}")

                    # Fallback condition is met
                    if uniform_sample_tangent_slope >= epsgreedy_tangent_slope:
                        print(f"Fallback strategy 2 at itr {itr} as uniformsample tangent is {uniform_sample_tangent_slope} and epsgreedy tangent is {epsgreedy_tangent_slope}")
                        # Change the algorithm to be uniform sample
                        algo_params = { 'type': 'UniformExploration' }
                        # Recreate the index into a flat, shuffled index
                        leaf_elements = index.get_leaf_elements()
                        index_dict = { 'children': leaf_elements }
                        index: Dict = get_index_from_dict(index_dict, True)
                        index.initialize_metadata('UniformExploration', algo_params)
                        fallback_switch_itr = -1
                    else:  # Fallback condition not met, check back in the next threshold
                        next_fallback_check_itr += int(n * fallback_params['frequency'])
            ### FALLBACK LOGIC END ###
            iter_fallback_check_time = time.time() - timestamp
            total_time_per_category['algo'] += iter_fallback_check_time
            time_elapsed_on_useful_work += iter_fallback_check_time


            timestamp = time.time()
            ### SAMPLING LOGIC START ###
            # Choose leaf node and sample IDs from it
            sample_ids, leaf_is_now_empty, selected_leaf_idx = select_leaf_and_sample(index, algo_params, itr, pq.kth_best_score(), sample_method, batch_size)
            realized_batch_size: int = len(sample_ids)  # If leaf has few elements left, this may be less than batch_size
            ### SAMPLING LOGIC END ###
            iter_sampling_time = time.time() - timestamp
            total_time_per_category['algo'] += iter_sampling_time
            time_elapsed_on_useful_work += iter_sampling_time


            timestamp = time.time()
            ### SCORING LOGIC START ###
            # Get the scores of the sample_ids
            scores: List[float] = score_sample_ids(sample_ids, sampling_fn, sampling_params, scoring_fn, scoring_params, skip_scoring_fn, gt_scores)
            realized_batch_size_2 = min(realized_batch_size, len(scores))
            ### SCORING END ###
            iter_scoring_time = time.time() - timestamp
            total_time_per_category['scorer'] += iter_scoring_time
            time_elapsed_on_useful_work += iter_scoring_time


            timestamp = time.time()
            ### PRIORITY QUEUE HANDLING START ###
            # For each score, update priority queue & handle logging
            for idx in range(realized_batch_size_2):
                # Update priority queue
                pq.insert(sample_ids[idx], scores[idx])
            ### PRIORITY QUEUE HANDLING END ###
            iter_pq_time = time.time() - timestamp
            total_time_per_category['pq'] += iter_pq_time
            time_elapsed_on_useful_work += iter_pq_time


            timestamp = time.time()
            ### INDEX METADATA UPDATE START ###
            for idx in range(realized_batch_size_2):
                # Update metadata over the index_builder
                index.update(algo_params['type'], selected_leaf_idx, scores[idx], pq.kth_best_score())
            ### INDEX METADATA UPDATE END ###
            iter_index_update_time = time.time() - timestamp
            total_time_per_category['algo'] += iter_index_update_time
            time_elapsed_on_useful_work += iter_index_update_time


            # Increment iteration number
            itr += realized_batch_size
            pbar.update(realized_batch_size)


            timestamp = time.time()
            ### EMPTY LEAF CLEANUP START ###
            # If the sample exhausted the leaf, then it needs to be cleaned up, including any recursively emptied parent
            if leaf_is_now_empty:
                subtract = algo_params['subtract'] if algo_params['type'] == 'EpsGreedy' else False
                clean_empty_leaf(index, selected_leaf_idx, subtract)
            ### EMPTY LEAF CLEANUP END ###
            iter_empty_leaf_cleanup_time = time.time() - timestamp
            total_time_per_category['algo'] += iter_empty_leaf_cleanup_time
            time_elapsed_on_useful_work += iter_empty_leaf_cleanup_time


            # Accumulate iteration result
            pq_elements, pq_scores = pq.get_heap()
            iter_result = get_iter_result(pq_elements, pq_scores, gt_rankings, gt_solution, time_elapsed_on_useful_work, k)
            for _ in range(realized_batch_size):
                iter_results.append(iter_result)


    ### FINAL RESULT AGGREGATION ###

    total_result = {
        "iter_results": iter_results,
        "solution_set": pq.get_heap()[0],
        "overhead_one_time": total_time_per_category['once'],
        "overhead_pq": total_time_per_category['pq'] / itr,
        "overhead_algo": total_time_per_category['algo'] / itr,
        "overhead_scorer": total_time_per_category['scorer'] / itr,
        "overhead_other": total_time_per_category['other'] / itr,
        "fallback_switch_itr": fallback_switch_itr
    }

    return total_result


def get_iter_result(pq_elements, pq_scores, gt_rankings, gt_solution, time, k) -> Dict:
    # Accumulate iteration result

    iter_result = {  # Statistics gathered by all runs
        "STK": sum(pq_scores),
        "KLS": pq_scores[-1] if len(pq_scores) >= k else 0.0,
        "time": time,
    }
    if gt_rankings is not None and gt_solution is not None:
        gt_k = len(gt_solution)
        iter_result["Precision@K"] = len([elem for elem in pq_elements if elem in gt_solution]) / gt_k
        iter_result["Recall@K"] = len([elem for elem in gt_solution if elem in pq_elements]) / gt_k
        iter_result["AvgRank"] = (np.sum(np.array([gt_rankings[str(elem)] for elem in pq_elements])) + max(k - len(pq_scores), 0) * len(gt_rankings)) / gt_k
        iter_result["WorstRank"] = len(gt_rankings) if len(pq_scores) < k else gt_rankings[str(pq_elements[-1])]

    return iter_result


def select_leaf_arm(index: IndexNode, algorithm: str, algo_params: Dict, itr: int, kth_best_score: float) -> List[int]:
    """
    Select a leaf node to sample from based on the bandit algorithm.

    :param index: IndexNode object representing the index_builder structure.
    :param algorithm: The bandit algorithm to use.
    :param algo_params: A dictionary of parameters to pass to the bandit algorithm.
    :param kth_best_score: The score of the k-th best sample in the priority queue.
    :param itr: The current iteration number.
    :return: The index_builder of the leaf node to sample from.
    """
    if algorithm == "UniformExploration":
        return select_leaf_arm_uniform(index)
    elif algorithm == "UCB":
        return select_leaf_arm_ucb(index, algo_params, itr)
    elif algorithm == "EpsGreedy":
        return select_leaf_arm_epsgreedy(index, algo_params, itr, kth_best_score)
    elif algorithm == "Scan":
        return select_leaf_arm_scan(index)
    else:
        raise ValueError("Unknown bandit algorithm.")

def select_leaf_arm_uniform(node: Union[IndexNode, IndexLeaf]) -> List[int]:
    """
    Select a leaf node to sample from uniformly at random.
    :param node: IndexNode or IndexLeaf object representing the current node.
    :return: A list of indices representing the path to the selected leaf node.
    """
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        num_children: int = len(node.children)
        child_idx: int = random.choice(range(num_children))
        children_arm: List[int] = select_leaf_arm_uniform(node.get_child_at(child_idx))
        return [child_idx] + children_arm

def select_leaf_arm_scan(node: Union[IndexNode, IndexLeaf]) -> List[int]:
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        return [0] + select_leaf_arm_scan(node.get_child_at(0))

def clean_empty_leaf(root, index_list, subtract_child: bool = True):
    """
    Deletes the leaf node specified by index_list from the tree rooted at root.
    Recursively deletes parent nodes if they become empty after deletion.
    Returns the new root (may be None if root is deleted).
    """
    # Stack to keep track of (parent_node, child_index)
    stack = []
    current = root
    for idx in index_list:
        stack.append((current, idx))
        try:
            current = current.children[idx]
        except IndexError:
            raise ValueError("Invalid index path")

    # Now 'current' is the leaf node to delete
    # Remove it from its parent's 'children' list
    while stack:
        parent, idx = stack.pop()
        if "histogram" in parent.metadata and subtract_child:
            parent.metadata['histogram'].subtract(parent.get_child_at(idx).metadata['histogram'])
        del parent.children[idx]
        if parent.children:
            # Parent still has children, stop recursion
            return root
        # If parent has no children, continue to delete it in the next iteration

    # If the root has no children after deletion, return None
    return None

def select_leaf_arm_ucb(node: IndexNode, algo_params: Dict, itr: int) -> List[int]:
    """
    Select a leaf node to sample from based on the UCB algorithm.
    :param node: IndexNode object representing the current node.
    :param algo_params: A dictionary of parameters to pass to the UCB algorithm.
    :param itr: The current iteration number.
    :return: A list of indices representing the path to the selected leaf node.
    """
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        children: List = node.children
        max_ucb: float = 0.0
        best_child_idx: int = 0
        # Find the child with the highest UCB value
        for idx, child in enumerate(children):
            if isinstance(child, IndexLeaf) and child.remaining_size() <= 0:
                upper_bound = 0.0
            # The UCB formula
            else:
                upper_bound: float = child.metadata['mean'] + algo_params['c'] * math.sqrt(math.log(itr) / node.metadata['count'])
            if upper_bound > max_ucb:  # Update the best child
                best_child_idx = idx
                max_ucb = upper_bound
        # Recurse down the tree
        return [best_child_idx] + select_leaf_arm_ucb(node.get_child_at(best_child_idx), algo_params, itr)

def select_leaf_arm_epsgreedy(node: IndexNode, algo_params: Dict, itr: int, kth_best_score: float) -> List[int]:
    """
    Select a leaf node to sample from based on the histogram epsilon-greedy algorithm.
    :param node: IndexNode object representing the current node.
    :param algo_params: A dictionary of parameters to pass to the epsilon-greedy algorithm.
    :param itr: The current iteration number.
    :param kth_best_score: The score of the k-th best sample in the priority queue.
    :return: A list of indices representing the path to the selected leaf node.
    """
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        # Decide whether to explore or exploit
        alpha = algo_params['alpha']
        eps = alpha * math.pow(itr / 25.0, -1.0/3.0)
        rand = random.random()
        if rand > eps:  # Exploitation
            children = node.children
            child_gains = [0.0 for _ in range(len(children))]
            # Find the child with the highest expected marginal gain
            for idx, child in enumerate(children):
                if isinstance(child, IndexLeaf) and child.remaining_size() == 0:
                    continue
                else:
                    gain = child.metadata['histogram'].expected_marginal_gain(kth_best_score)
                    child_gains[idx] = gain
            max_gain = max(child_gains)
            best_children_idxs = [idx for idx, child in enumerate(children) if child_gains[idx] >= max_gain]
            best_child_idx = random.choice(best_children_idxs)
            greedy_arm = [best_child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(best_child_idx), algo_params, itr, kth_best_score)
            return greedy_arm
        else:  # Exploration
            num_children: int = len(node.children)
            child_idx: int = random.choice(range(num_children))
            arm = [child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(child_idx), algo_params, itr, kth_best_score)
            return arm

def termination_condition_is_met(budget: Union[int, float], itr: int) -> bool:
    if isinstance(budget, int):
        if itr > budget:
            return True
    elif isinstance(budget, float):
        time_now = time.time()
        if time_now - time_start >= budget:
            return True
    else:
        raise ValueError("Given budget type is invalid, must be int (itrs) or float (time).")
    return False

def get_initialized_index(index_filename, sample_method, algo_params) -> Dict:
    # Load index from file
    with open(Path.cwd() / index_filename, 'r') as file:
        index_dict: Dict = json.load(file)

    if sample_method == "reversescan":
        index_dict["children"].reverse()

    # Shuffle the clusters in the index if sample method is random, otherwise leave insertion order
    shuffle_elements_in_index: bool = False if (sample_method == "scan" or sample_method == "sortedscan" or sample_method == "reversescan") else True
    index: IndexNode = get_index_from_dict(index_dict, shuffle_elements_in_index)

    index.initialize_metadata(algo_params['type'], algo_params)

    return index

def select_leaf_and_sample(index: IndexNode, algo_params: Dict, itr: int, kth_best_score: float, sample_method: str,
                           batch_size: int) -> Tuple[List, bool]:
    selected_leaf_idx: List[int] = select_leaf_arm(index, algo_params['type'], algo_params, itr, kth_best_score)
    leaf_node: IndexLeaf = index.get_grandchild(selected_leaf_idx)
    sample_ids: List
    leaf_is_now_empty: bool
    if sample_method == "replace":
        sample_ids = leaf_node.sample_with_replacement(batch_size)
        leaf_is_now_empty = False
    elif sample_method == "noreplace" or sample_method == "scan" or sample_method == "sortedscan":
        sample_ids, leaf_is_now_empty = leaf_node.sample_without_replacement(batch_size)
    else:
        raise ValueError
    return sample_ids, leaf_is_now_empty, selected_leaf_idx

def score_sample_ids(sample_ids: List, sampling_fn: Callable, sampling_params: Dict, scoring_fn: Callable, scoring_params: Dict, skip_scoring_fn: bool, gt_scores: Dict) -> List[float]:
    if not skip_scoring_fn:
        objects: List[Any] = sampling_fn(sample_ids, sampling_params)
        if len(objects) == 0:
            return []
        scores: List[float] = scoring_fn(objects, scoring_params)
    else:
        scores: List[float] = [gt_scores[idx] for idx in sample_ids]
    return scores

