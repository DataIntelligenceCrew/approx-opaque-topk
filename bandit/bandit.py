import json
import math
import random
import time
from typing import List, Callable, Union, Dict, Set

import numpy as np

from bandit.index import IndexNode, IndexLeaf, get_index_from_dict
from bandit.limited_pq import LimitedPQ


def approx_top_k_bandit(index_params: Dict, k: int, scoring_fn: Callable, scoring_params: Dict, sampling_fn: Callable,
                        sampling_params: Dict, algorithm: str, algo_params: Dict, budget: Union[int, float],
                        sample_method: str, batch_size: int, gt_rankings: Dict = None, gt_solution: Set[str] = None) -> Dict:
    """
    Perform a run of approximate top-k bandit algorithm.
    This assumes that the index_builder has already been loaded, but not initialized with bandit-related metadata.

    :param index_params: Location of JSON index file.
    :param k: Cardinality constraint for top-k query.
    :param scoring_fn: A function which takes a sample and returns a score.
    :param scoring_params: A dictionary of parameters to pass to the scoring function.
    :param sampling_fn: A function which takes a string sample identifier and returns a sample.
    :param sampling_params: A dictionary of parameters to pass to the sampling function.
    :param algorithm: The bandit algorithm to use.
    :param budget: The budget for the bandit run. This can be either an integer (number of iterations) or a float (time in seconds).
    :param algo_params: A dictionary of parameters to pass to the bandit algorithm.
    :param sample_method: The method for sampling from the leaf nodes. Either "scan", "replace", or "noreplace".
    :param batch_size: Sample up to this number of elements from the cluster at a time.
    :param gt_rankings: A dictionary mapping from string IDs to ground truth rankings.
    :param gt_solution: The optimal ground truth solution containing k highest-scoring elements in the dataset.
    :return: A dictionary representing the result of this bandit run. It has the following fields:
             - "iter_results": A list of results for each iteration. Each iteration's result is a dictionary with the
                               following statistics:
                - Always: STK, KLS, time, element, score
                - Only if gt_rankings and gt_solution were provided: Precision@K, Recall@K, AvgRank, WorstRank
             - "solution_set": A list of string IDs for the final solution.
             - "index_time": Time it took for the initial index construction overhead.
             - "iter_time": Average time it took for each iteration.
    """
    # For some scoring functions, preprocessing params is necessary
    index_start_time = time.time()

    # Load index from file
    with open(index_params['file'], 'r') as file:
        index_dict = json.load(file)

    # Shuffle the clusters in the index if sample method is random, otherwise leave insertion order
    index = get_index_from_dict(index_dict, False if sample_method == "scan" else True)
    index.initialize_metadata(algorithm, algo_params)
    index_end_time = time.time()
    index_time = index_end_time - index_start_time

    # Initialize bookkeeping index_metadata structures
    time_start = time.time()
    itr = 1
    pq: LimitedPQ = LimitedPQ(k)
    iter_results: List[Dict] = []
    running_solution = None

    # Main loop
    while True:
        # Check termination condition
        if isinstance(budget, int):
            if itr > budget:
                break
        elif isinstance(budget, float):
            time_now = time.time()
            if time_now - time_start >= budget:
                break
        else:
            raise ValueError("Budget must be int (iterations) or float (time in seconds).")

        # Check if index is empty
        if index.children is None or len(index.children) == 0:
            break

        # Choose leaf node and sample from it
        selected_leaf_idx: List[int] = select_leaf_arm(index, algorithm, algo_params, itr, pq.kth_best_score())
        leaf_node: IndexLeaf = index.get_grandchild(selected_leaf_idx)
        iter_batch_size = batch_size
        if sample_method == "replace":
            sample_ids = leaf_node.sample_with_replacement(iter_batch_size)
            leaf_is_now_empty = False
        elif sample_method == "noreplace" or sample_method == "scan":
            sample_ids, leaf_is_now_empty = leaf_node.sample_without_replacement(iter_batch_size)
        else:
            raise ValueError

        # Obtain the actual objects for the sampled IDs and score them
        inputs_to_scorers = sampling_fn(sample_ids, sampling_params)
        scores: List[float] = scoring_fn(inputs_to_scorers, scoring_params)

        # For each score, update priority queue & handle logging
        for idx in range(len(scores)):
            sample_id = sample_ids[idx]
            sample_score = scores[idx]

            # Update priority queue
            pq.insert(sample_id, sample_score)
            # Update metadata over the index_builder
            index.update(algorithm, selected_leaf_idx, sample_score, pq.kth_best_score())

            # Accumulate iteration result
            pq_elements, pq_scores = pq.get_heap()
            iter_result = get_iter_result(sample_id, sample_score, pq_elements, pq_scores, gt_rankings, gt_solution, time.time() - time_start, k)
            iter_results.append(iter_result)

            running_solution = pq_elements

            # Increment iteration
            itr += 1
        #print(itr)

        # If the sample exhausted the leaf, then it needs to be cleaned up, including any recursively emptied parent
        if leaf_is_now_empty:
            subtract = algo_params['subtract'] if algorithm == 'EpsGreedy' else False
            clean_empty_leaf(index, selected_leaf_idx, subtract)

    total_result = {
        "iter_results": iter_results,
        "solution_set": running_solution,
        "index_time": index_time,
        "iter_time": iter_results[-1]["time"] / len(iter_results)
    }
    return total_result


def get_iter_result(sample_id, sample_score, pq_elements, pq_scores, gt_rankings, gt_solution, time, k) -> Dict:
    # Accumulate iteration result

    iter_result = {  # Statistics gathered by all runs
        "STK": sum(pq_scores),
        "KLS": pq_scores[-1] if len(pq_scores) >= k else 0.0,
        "time": time,
        "element": sample_id,
        "score": sample_score
    }
    if gt_rankings is not None and gt_solution is not None:
        gt_k = len(gt_solution)
        iter_result["Precision@K"] = len([elem for elem in pq_elements if elem in gt_solution]) / gt_k
        iter_result["Recall@K"] = len([elem for elem in gt_solution if elem in pq_elements]) / gt_k
        iter_result["AvgRank"] = np.sum(np.array([gt_rankings[str(elem)] for elem in pq_elements])) / gt_k
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
            max_gain = 0.0
            best_child_idx = 0
            # Find the child with the highest expected marginal gain
            for idx, child in enumerate(children):
                if isinstance(child, IndexLeaf) and child.remaining_size() == 0:
                    gain = 0.0
                else:
                    gain = child.metadata['histogram'].expected_marginal_gain(kth_best_score)
                if gain > max_gain:  # Update the best child
                    best_child_idx = idx
                    max_gain = gain
            greedy_arm = [best_child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(best_child_idx), algo_params, itr, kth_best_score)
            return greedy_arm
        else:  # Exploration
            num_children: int = len(node.children)
            child_idx: int = random.choice(range(num_children))
            arm = [child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(child_idx), algo_params, itr, kth_best_score)
            return arm
