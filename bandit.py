import json
import math
from min_max_heap import MinMaxHeap
import random
import time
from typing import List, Callable, Union, Tuple, Dict, Self

class Histogram:
    def __init__(self, bin_borders: List[float], rebin_decay: float = 0.95, enlarge_factor: float = 1.2):
        self._num_bins: int = len(bin_borders) - 1
        self._bin_borders: List[float] = bin_borders
        self._bin_counts: List[float] = [0.0 for _ in range(self._num_bins)]
        self._total_counts: float = 0.0
        self._rebin_decay: float = rebin_decay
        self._enlarge_factor: float = enlarge_factor

    def expected_marginal_gain(self, kth_largest_score: float):
        gain = 0.0
        for b in range(self._num_bins):
            lo = max(kth_largest_score, self._bin_borders[b])
            hi = self._bin_borders[b + 1]
            if hi > lo:
                bin_gain = (self._bin_counts[b] / self._total_counts) * (hi + lo) / 2.0
                gain += bin_gain
        return gain

    def rebin(self, new_bin_borders: List[float]) -> Self:
        new_histogram = Histogram(new_bin_borders, rebin_decay=self._rebin_decay, enlarge_factor=self._enlarge_factor)
        new_bin_counts = [0.0 for _ in range(new_histogram._num_bins)]
        for b_old in range(self._num_bins):
            for b_new in range(new_histogram._num_bins):
                lo = max(self._bin_borders[b_old], new_bin_borders[b_new])
                hi = min(self._bin_borders[b_old+1], new_bin_borders[b_new+1])
                if hi > lo:
                    new_bin_counts += self._bin_counts[b_old] * (hi - lo) / (self._bin_borders[b_old+1] - self._bin_borders[b_old])
        new_bin_counts = [self._rebin_decay * x for x in new_bin_counts]
        new_total_counts = sum(new_bin_counts)
        new_histogram._bin_counts = new_bin_counts
        new_histogram._total_counts = new_total_counts
        return new_histogram

    def update_from_score(self, score: float) -> Self:
        histogram = self
        range_max = self._bin_borders[-1]
        if score > range_max:
            new_borders = Histogram._uniformly_divide_range_with_large_first_bin(histogram._bin_borders[0], histogram._bin_borders[1], histogram._bin_borders[-1], histogram._num_bins)
            histogram = histogram.rebin(new_borders)
        if histogram._bin_borders[1] > histogram._bin_borders[2]:
            new_borders = Histogram._uniformly_divide_range_with_large_first_bin(histogram._bin_borders[0], histogram._bin_borders[2], histogram._bin_borders[-1], histogram._num_bins)
            histogram = histogram.rebin(new_borders)
        for b in range(histogram._num_bins):
            b_lo = histogram._bin_borders[b]
            if score > b_lo:
                histogram._bin_counts[b] += 1.0
                break
        histogram._total_counts += 1.0
        return histogram

    @staticmethod
    def uniformly_divide_range(min_value: float, max_value: float, num_bins: int) -> List[float]:
        return [min_value + (max_value - min_value) * (x / num_bins) for x in range(num_bins + 1)]

    @staticmethod
    def _uniformly_divide_range_with_large_first_bin(min_value: float, first_bin_hi: float, max_value: float, num_bins: int) -> List[float]:
        return [min_value] + Histogram.uniformly_divide_range(first_bin_hi, max_value, num_bins - 1)


class IndexLeaf:
    def __init__(self, children: List[str], metadata: Dict):
        """
        :param children: A list of IDs for the elements that belong to this leaf node, where IDs are given as strings.
        """
        self._n: int = len(children)  # The total number of children under this leaf node
        self._children: List[str] = children  # Keep track of the children's IDs
        random.shuffle(self._children)  # Randomly shuffle the children for sampling purposes
        self._next_sample_idx: int = 0  # Use a simple integer for walking through the shuffle
        self.metadata = metadata  # Any information attached to this node for bandit

    def sample_without_replacement(self) -> Union[str, None]:
        """
        :return: Either the ID of a randomly sampled child, or None if no unseen child remains.
        """
        if self._next_sample_idx >= self._n:  # No unseen child remains
            return None
        else:  # There are unseen children left
            sample = self._children[self._next_sample_idx]
            self._next_sample_idx += 1
            return sample

    def sample_with_replacement(self) -> Union[str, None]:
        """
        :return: The ID of a randomly sampled child, or None if there is no child.
        """
        return random.choice(self._children)

    def total_size(self) -> int:
        """
        :return: The total number of children, at initialization time, under this leaf node.
        """
        return self._n

    def remaining_size(self) -> int:
        """
        :return: The number of unseen children under this leaf node.
        """
        return max(self._n - self._next_sample_idx, 0)

    def get_grandchild(self, grandchild_idx: List[int]) -> Self:
        if len(grandchild_idx) > 0:
            raise ValueError("Leaf node got a non-empty grandchild index.")
        return self

    def to_dict(self) -> Dict:
        return {"children": None, "metadata": self.metadata}

    def _initialize_ucb_metadata(self, params: Dict):
        init = params['init']
        self.metadata.update({'mean': init, 'count': 0.0, 'ucb': init})

    def _initialize_epsgreedy_metadata(self, params: Dict):
        bin_borders = Histogram.uniformly_divide_range(params['min'], params['max'], params['num_bins'])
        histogram = Histogram(bin_borders, params['rebin_decay'], params['enlarge_factor'])
        self.metadata.update({'histogram': histogram})

    def update_ucb(self, selected_leaf_idx: List[int], score: float, itr: int):
        old_mean = self.metadata['mean']
        old_cnt = self.metadata['count']
        new_cnt = old_cnt + 1
        new_mean = old_mean * old_cnt / new_cnt + score / new_cnt
        new_ucb = new_mean + math.sqrt(math.log(new_cnt) / (itr + 1))
        self.metadata.update({'mean': new_mean, 'count': new_cnt, 'ucb': new_ucb})

    def update_epsgreedy(self, selected_leaf_idx: List[int], score: float):
        old_histogram: Histogram = self.metadata['histogram']
        new_histogram = old_histogram.update_from_score(score)
        self.metadata.update({'histogram': new_histogram})


class IndexNode:
    def __init__(self, children: Union[List[Self], List[IndexLeaf]], metadata = None):
        self.children: Union[List[Self], List[IndexLeaf]] = children  # A list of 0-indexed children
        self.metadata = metadata  # Any additional information being stored at this node

    def get_child_at(self, child_idx: int) -> Union[Self, IndexLeaf]:
        return self.children[child_idx]

    def get_grandchild(self, grandchild_idx: List[int]) -> Union[Self, IndexLeaf]:
        child = self.get_child_at(grandchild_idx[0])
        return child.get_grandchild(grandchild_idx[1:])

    def to_dict(self) -> Dict:
        return {"children": [child.to_dict() for child in self.children], "metadata": self.metadata}

    def initialize_metadata(self, algorithm: str, params: Dict):
        match algorithm:
            case "UniformExploration":
                pass
            case "UCB":
                self._initialize_ucb_metadata(params)
            case "EpsGreedy":
                self._initialize_epsgreedy_metadata(params)

    def _initialize_ucb_metadata(self, params: Dict):
        init = params['init']
        self.metadata.update({'mean': init, 'count': 0.0, 'ucb': init})
        for child in self.children:
            child._initialize_ucb_metadata(params)

    def _initialize_epsgreedy_metadata(self, params: Dict):
        bin_borders = Histogram.uniformly_divide_range(params['min'], params['max'], params['num_bins'])
        histogram = Histogram(bin_borders, params['rebin_decay'], params['enlarge_factor'])
        self.metadata.update({'histogram': histogram})
        for child in self.children:
            child._initialize_epsgreedy_metadata(params)

    def update(self, algorithm: str, selected_leaf_idx: List[int], score: float, itr: int):
        match algorithm:
            case "UniformExploration":
                pass
            case "UCB":
                self.update_ucb(selected_leaf_idx, score, itr)
            case "EpsGreedy":
                self.update_epsgreedy(selected_leaf_idx, score)

    def update_ucb(self, selected_leaf_idx: List[int], score: float, itr: int):
        old_mean = self.metadata['mean']
        old_cnt = self.metadata['count']
        new_cnt = old_cnt + 1
        new_mean = old_mean * old_cnt / new_cnt + score / new_cnt
        new_ucb = new_mean + math.sqrt(math.log(new_cnt) / (itr + 1))
        self.metadata.update({'mean': new_mean, 'count': new_cnt, 'ucb': new_ucb})
        child = self.get_child_at(selected_leaf_idx[0])
        child.update_ucb(selected_leaf_idx[1:], score, itr)

    def update_epsgreedy(self, selected_leaf_idx: List[int], score: float):
        old_histogram: Histogram = self.metadata['histogram']
        new_histogram = old_histogram.update_from_score(score)
        self.metadata.update({'histogram': new_histogram})
        child = self.get_child_at(selected_leaf_idx[0])
        child.update_epsgreedy(selected_leaf_idx[1:], score)


def store_index_to_json(index: IndexNode, filename: str):
    index_dict = index.to_dict()
    with open (filename, 'w') as json_file:
        json.dump(index_dict, json_file)

def load_index_from_json(filename: str) -> IndexNode:
    with open (filename, 'r') as json_file:
        return json.load(json_file)

class LimitedPQ:
    def __init__(self, cardinality_constraint: int):
        self.k = cardinality_constraint
        self.heap = MinMaxHeap()

    def insert(self, element, score):
        self.heap.push((-score, element))
        if len(self.heap) > self.k:
            self.heap.pop_max()

    def get_heap(self) -> List[Tuple[float, str]]:
        return self.heap.queue

    def kth_best_score(self) -> float:
        return -1 * self.heap.peek_min()

def approx_top_k_bandit(index: IndexNode, k: int, scoring_fn: Callable, sampling_fn: Callable, sampling_params: Dict, algorithm: str, budget: Union[int, float], algo_params: Dict = None) -> List:
    # Initialize bookkeeping data structures
    index.initialize_metadata(algorithm, algo_params)
    time_start = time.time()
    itr = 0
    pq: LimitedPQ = LimitedPQ(k)
    print("[")
    while True:
        # Check termination condition
        if isinstance(budget, int):
            if itr >= budget:
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
        sample = sampling_fn(sample_id, **sampling_params)
        score: float = scoring_fn(sample)
        # Update priority queue
        pq.insert(sample, score)
        # Update metadata over the index
        index.update(algorithm, selected_leaf_idx, score, itr)
        # Print out iteration result
        result = {
            "iteration": itr,
            "arm": selected_leaf_idx,
            "sample_id": sample_id,
            "score": score,
            "pq": pq.get_heap(),
            "index": index.to_dict()
        }
        print(json.dumps(result) + ",")
    print("]")
    return pq.get_heap()


def select_leaf_arm(index: IndexNode, algorithm: str, params: Dict, kth_best_score: float, itr: int) -> List[int]:
    match algorithm:
        case "UniformExploration":
            return select_leaf_arm_uniform(index)
        case "UCB":
            return select_leaf_arm_ucb(index)
        case "EpsGreedy":
            return select_leaf_arm_epsgreedy(index, kth_best_score, params, itr)

def select_leaf_arm_uniform(node: Union[IndexNode, IndexLeaf]) -> List[int]:
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        num_children: int = len(node.children)
        child_idx: int = random.choice(range(num_children))
        children_arm: List[int] = select_leaf_arm_uniform(node.get_child_at(child_idx))
        return [child_idx] + children_arm

def select_leaf_arm_ucb(node: IndexNode) -> List[int]:
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        children = node.children
        max_ucb = 0.0
        best_child_idx = 0
        for child, idx in enumerate(children):
            upper_bound = child.metadata['ucb']
            if upper_bound > max_ucb:
                best_child_idx = idx
        return [best_child_idx] + select_leaf_arm_ucb(node.get_child_at(best_child_idx))

def select_leaf_arm_epsgreedy(node: IndexNode, kth_best_score: float, params: Dict, itr: int) -> List[int]:
    if isinstance(node, IndexLeaf):
        return []
    elif isinstance(node, IndexNode):
        alpha = params['alpha']
        eps = alpha * math.pow(itr, -1.0/3.0)
        rand = random.random()
        if rand < eps:
            children = node.children
            max_gain = 0.0
            best_child_idx = 0
            for idx, child in enumerate(children):
                gain = child.metadata['histogram'].expected_marginal_gain(kth_best_score)
                if gain > max_gain:
                    best_child_idx = idx
            return [best_child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(best_child_idx), kth_best_score, params, itr)
        else:
            num_children: int = len(node.children)
            child_idx: int = random.choice(range(num_children))
            return [child_idx] + select_leaf_arm_epsgreedy(node.get_child_at(child_idx), kth_best_score, params, itr)
