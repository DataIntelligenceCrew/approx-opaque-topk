import math
import random
import time
from typing import List, Callable, Union, Tuple, Dict, Self

from bandit.index import IndexNode, IndexLeaf
from bandit.limited_pq import LimitedPQ


def approx_top_k_bandit(index: IndexNode, k: int, scoring_fn: Callable, scoring_params: Dict, sampling_fn: Callable,
                        sampling_params: Dict, algorithm: str, algo_params: Dict, budget: Union[int, float]) -> List[Dict]:
    """
    Perform a run of approximate top-k bandit algorithm.
    This assumes that the index has already been loaded, but not initialized with bandit-related metadata.

    :param index: IndexNode object representing the index structure.
    :param k: Cardinality constraint for top-k query.
    :param scoring_fn: A function which takes a sample and returns a score.
    :param scoring_params: A dictionary of parameters to pass to the scoring function.
    :param sampling_fn: A function which takes a string sample identifier and returns a sample.
    :param sampling_params: A dictionary of parameters to pass to the sampling function.
    :param algorithm: The bandit algorithm to use.
    :param budget: The budget for the bandit run. This can be either an integer (number of iterations) or a float (time in seconds).
    :param algo_params: A dictionary of parameters to pass to the bandit algorithm. The required fields are:
                        - Nothing for UniformExploration.
                        - 'c' for UCB. If c is large, the algorithm will be more exploratory.
                        - 'alpha' for EpsGreedy. If alpha is large, the algorithm will be more exploratory.
                        - 'max' for EpsGreedy. This is the initial estimate for maximum score.
                        - 'num_bins' for EpsGreedy. This is the number of bins for the histogram.
                        - 'rebin_decay' for EpsGreedy. This is the decay factor whenever the histogram is re-binned.
                        - 'enlarge_factor' for EpsGreedy. This is the factor by which the histogram is enlarged when the max score is updated.
                        - 'enlarge_lowest' for EpsGreedy. This controls whether the smallest bin enlarges as S_(k) increases.
    :return: A list of dictionaries, each representing the result of an iteration.
    """
    # Initialize bookkeeping data structures
    index.initialize_metadata(algorithm, algo_params)
    time_start = time.time()
    itr = 1
    pq: LimitedPQ = LimitedPQ(k)
    iter_results: List[Dict] = []
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
        # Choose leaf node and sample from it
        selected_leaf_idx: List[int] = select_leaf_arm(index, algorithm, algo_params, pq.kth_best_score(), itr)
        leaf_node: IndexLeaf = index.get_grandchild(selected_leaf_idx)
        sample_id: str = leaf_node.sample_with_replacement()
        sample = sampling_fn(sample_id, sampling_params)
        score: float = scoring_fn(sample, scoring_params)
        # Update priority queue
        pq.insert(sample, score)
        # Update metadata over the index
        index.update(algorithm, selected_leaf_idx, score, itr)
        # Accumulate iteration result
        result = {
            "iteration": itr,
            "arm": selected_leaf_idx,
            "sample_id": sample_id,
            "score": score,
            "pq": pq.get_heap(),
            "index": index.to_dict(),
            "time": time.time() - time_start
        }
        iter_results.append(result)
        # Increment iteration counter
        itr += 1
    return iter_results

def select_leaf_arm(index: IndexNode, algorithm: str, algo_params: Dict, kth_best_score: float, itr: int) -> List[int]:
    """
    Select a leaf node to sample from based on the bandit algorithm.

    :param index: IndexNode object representing the index structure.
    :param algorithm: The bandit algorithm to use.
    :param algo_params: A dictionary of parameters to pass to the bandit algorithm.
    :param kth_best_score: The score of the k-th best sample in the priority queue.
    :param itr: The current iteration number.
    :return: The index of the leaf node to sample from.
    """
    if algorithm == "UniformExploration":
        return select_leaf_arm_uniform(index)
    elif algorithm == "UCB":
        return select_leaf_arm_ucb(index, algo_params, itr)
    elif algorithm == "EpsGreedy":
        return select_leaf_arm_epsgreedy(index, algo_params, itr, kth_best_score)
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
            # The UCB formula
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
        eps = alpha * math.pow(itr, -1.0/3.0)
        rand = random.random()
        if rand < eps:  # Exploitation
            children = node.children
            max_gain = 0.0
            best_child_idx = 0
            # Find the child with the highest expected marginal gain
            for idx, child in enumerate(children):
                gain = child.metadata['histogram'].expected_marginal_gain(kth_best_score)
                if gain > max_gain:  # Update the best child
                    best_child_idx = idx
            return [best_child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(best_child_idx), algo_params, itr, kth_best_score)
        else:  # Exploration
            num_children: int = len(node.children)
            child_idx: int = random.choice(range(num_children))
            return [child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(child_idx), algo_params, itr, kth_best_score)
