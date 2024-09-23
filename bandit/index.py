import json
import random
from typing import List, Dict, Union, Tuple
from typing_extensions import Self

from bandit.histogram import Histogram

"""
The index_builder builder methods build the index_builder as a JSON file. 
During query execution, it is read into memory and teh whole index_builder stays in memory at all times. 
The index_builder index_metadata structure is built from a dictionary representation of the JSON index_builder. 
In addition to holding the dendrogram structure, it also holds information about which string identifiers belong to each
leaf node, as well as additional metadata per node for bandit algorithms. 
"""


class IndexLeaf:
    """
    IndexLeaf represents a leaf node in the index_builder.
    A leaf node is a cluster at the bottom of the index_builder structure that contains a list of string identifiers.
    The IDs are then used by the sampler methods to retrieve the actual index_metadata points from the index_metadata store.
    """

    def __init__(self, children: List[str], metadata: Dict = None, shuffle: bool = True):
        """
        Initializes a new IndexLeaf with given children and metadata.

        :param children: A list of identifiers for the elements that belong to this leaf node.
        """
        self._n: int = len(children)  # The total number of children under this leaf node
        self.children: List[str] = children  # Keep track of the children's IDs
        if shuffle:
            random.shuffle(self.children)  # Randomly shuffle the children for sampling purposes
        self._next_sample_idx: int = 0  # Use a simple integer for walking through the shuffle
        self.metadata: Dict = metadata if metadata is not None else dict()  # Any information attached to this node

    def sample_without_replacement(self, batch_size: int) -> Tuple[List[str], bool]:
        """
        Performs multiple samplings without replacement from the children of this leaf node.
        The sample walks through the shuffled list of children up to batch_size indices.

        :param batch_size: The number of samples to return.
        :return: The IDs of the randomly sampled children, and whether this sample made this leaf empty.
        """
        samples = []
        is_now_empty = False
        for _ in range(batch_size):
            sample, emptied = self.sample_one_without_replacement()
            if sample is not None:
                samples.append(sample)
            is_now_empty = is_now_empty or emptied
        return samples, is_now_empty

    def sample_one_without_replacement(self) -> Tuple[Union[str, None], bool]:
        """
        Performs sampling without replacement from the children of this leaf node.
        The sample is implemented as simply a walk through the shuffled list of children by an index_builder.

        :return: Either the ID of a randomly sampled child, or None if no unseen child remains; and if this sample made this leaf empty.
        """
        if self._next_sample_idx >= self._n:  # No unseen child remains
            return None, True
        else:  # There are unseen children left
            sample: str = self.children[self._next_sample_idx]
            self._next_sample_idx += 1
            return sample, self.remaining_size() <= 0

    def sample_with_replacement(self, batch_size: int) -> List[str]:
        """
        Performs sampling with replacement from ths children of this leaf node.
        The sample is chosen from the initial set of children.
        May behave unexpectedly if sample_without_replacement and sample_with_replacement are intermixed.

        :param batch_size: The number of samples to return.
        :return: The ID of a randomly sampled child, or None if there is no child.
        """
        return random.sample(self.children, batch_size)

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
        """
        Method used to obtain a leaf node from an index_builder specifying it (e.g. [0, 1, 2]).
        Since this is a leaf node, the grandchild index_builder should be empty.

        :param grandchild_idx: The index_builder specifying the leaf node to be retrieved.
        :return: The leaf node itself.
        """
        if len(grandchild_idx) > 0:
            raise ValueError("Leaf node got a non-empty grandchild index_builder.")
        return self

    def to_dict(self, include_children: bool = False) -> Dict:
        """
        :param include_children: Whether to include the (remaining) children in the dictionary representation.
        :return: A dictionary representation of the leaf node.
        """
        return {"children": self.children[self._next_sample_idx:] if include_children else None, "histogram": self.metadata['histogram'].to_dict()}

    def update(self, algorithm: str, selected_leaf_idx: List[int], score: float, kth_largest_score: float):
        """
        Updates the metadata of the leaf node and its ancestors based on the bandit algorithm used.

        :param algorithm: The bandit algorithm used for updating the metadata.
        :param selected_leaf_idx: The index_builder of the leaf node that was selected during the query.
        :param score: The reward obtained from the selected leaf node.
        :param kth_largest_score: The current S_(k) value.
        """
        if algorithm == "UniformExploration":
            pass
        elif algorithm == "UCB":
            self._update_ucb(selected_leaf_idx, score)
        elif algorithm == "EpsGreedy":
                self._update_epsgreedy(selected_leaf_idx, score, kth_largest_score)
        elif algorithm == "Scan":
            pass
        else:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

    def _update_ucb(self, selected_leaf_idx: List[int], score: float):
        # Get old count and mean
        old_mean: float = self.metadata['mean']
        old_cnt: float = self.metadata['count']
        # Compute new count and mean
        new_cnt: float = old_cnt + 1
        new_mean: float = old_mean * old_cnt / new_cnt + score / new_cnt
        self.metadata.update({'mean': new_mean, 'count': new_cnt})

    def _update_epsgreedy(self, selected_leaf_idx: List[int], score: float, kth_largest_score: float):
        old_histogram: Histogram = self.metadata['histogram']
        old_histogram.update_from_score(score, kth_largest_score)

    def initialize_metadata(self, algorithm: str, params: Dict):
        """
        Initializes the metadata of the leaf node based on the bandit algorithm used.

        :param algorithm: The bandit algorithm used for initializing the metadata.
        :param params: A dictionary containing the initialization parameters for the bandit algorithm.
        """
        if algorithm == "UniformExploration":
            pass
        elif algorithm == "UCB":
            self._initialize_ucb_metadata(params)
        elif algorithm == "EpsGreedy":
            self._initialize_epsgreedy_metadata(params)
        elif algorithm == "Scan":
            pass
        else:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

    def _initialize_ucb_metadata(self, params: Dict):
        """
        :param params: A dictionary which contains the initialization value for the mean ('init').
        """
        init = params['init']
        metadata = {
            'mean': init,  # The mean value of the rewards seen so far
            'count': 0.0,  # The number of rewards seen so far
        }
        self.metadata.update(metadata)

    def _initialize_epsgreedy_metadata(self, params: Dict):
        """
        :param params: A dictionary which contains the parameters for the histogram ('min', 'max', 'num_bins',
                       'rebin_decay', 'enlarge_factor', 'enlarge_lowest').
        """
        histogram = Histogram.new_empty_uniform(params['min'], params['max'], params['num_bins'], params['rebin_decay'],
                                                params['enlarge_max_factor'], params['enlarge_lowest'])
        self.metadata.update({'histogram': histogram})


class IndexNode:
    """
    IndexNode represents any non-leaf node in the index_builder.
    An IndexNode has a list of children, which can be either other IndexNodes or IndexLeaves.
    """

    def __init__(self, children: List[Union[Self, IndexLeaf]], metadata = None):
        """
        Initializes a new IndexNode with given children and metadata.

        :param children: A list of children nodes, which can be either IndexNodes or IndexLeaves.
        :param metadata: Any additional information being stored at this node.
        """
        self.children: Union[List[Self], List[IndexLeaf]] = children  # A list of 0-indexed children
        self.metadata = metadata if metadata is not None else dict()  # Any additional information stored for this node

    def get_child_at(self, child_idx: int) -> Union[Self, IndexLeaf]:
        """
        :param child_idx: The index_builder of the child to be retrieved.
        :return: The child node at the specified index_builder.
        """
        return self.children[child_idx]

    def get_grandchild(self, grandchild_idx: List[int]) -> Union[Self, IndexLeaf]:
        """
        Method used to obtain a leaf node from an index_builder specifying it (e.g. [0, 1, 2]).
        The method recursively calls itself on the children of the current node.

        :param grandchild_idx: The index_builder specifying the leaf node to be retrieved.
        :return: The leaf grandchild node.
        """
        child: Union[Self, IndexLeaf] = self.get_child_at(grandchild_idx[0])
        return child.get_grandchild(grandchild_idx[1:])

    def to_dict(self) -> Dict:
        """
        :return: A dictionary representation of the index_builder node.
        """
        return {"children": [child.to_dict() for child in self.children], "metadata": self.metadata}

    def initialize_metadata(self, algorithm: str, params: Dict):
        if algorithm == "UniformExploration":
            pass
        elif algorithm == "UCB":
            self._initialize_ucb_metadata(params)
        elif algorithm == "EpsGreedy":
            self._initialize_epsgreedy_metadata(params)
        elif algorithm == "Scan":
            pass
        else:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

    def _initialize_ucb_metadata(self, params: Dict):
        init = params['init']
        # Initializing count as 1 adds initial bias but prevents division by zero
        self.metadata.update({'mean': init, 'count': 0.001})
        for child in self.children:
            child._initialize_ucb_metadata(params)

    def _initialize_epsgreedy_metadata(self, params: Dict):
        # Create histogram and save it to metadata
        histogram = Histogram.new_empty_uniform(0.0, params['max'], params['num_bins'], params['rebin_decay'],
                                                params['enlarge_max_factor'], params['enlarge_lowest'])
        self.metadata.update({'histogram': histogram})
        for child in self.children:
            child._initialize_epsgreedy_metadata(params)

    def update(self, algorithm: str, selected_leaf_idx: List[int], score: float, kth_largest_score: float):
        """
        Update the metadata of a leaf node and its ancestors based on the bandit algorithm used.

        :param algorithm: The bandit algorithm used for updating the metadata.
        :param selected_leaf_idx: The index_builder of the leaf node that was selected during the query.
        :param score: The reward obtained from the selected leaf node.
        :param kth_largest_score: The current S_(k) value.
        """
        if algorithm == "UniformExploration":
            pass
        elif algorithm == "UCB":
            self.update_ucb(algorithm, selected_leaf_idx, score, kth_largest_score)
        elif algorithm == "EpsGreedy":
            self.update_epsgreedy(algorithm, selected_leaf_idx, score, kth_largest_score)
        elif algorithm == "Scan":
            pass
        else:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

    def update_ucb(self, algorithm: str, selected_leaf_idx: List[int], score: float, kth_largest_score: float):
        old_mean: float = self.metadata['mean']
        old_cnt: float = self.metadata['count']
        new_cnt: float = old_cnt + 1
        new_mean: float = old_mean * old_cnt / new_cnt + score / new_cnt
        self.metadata.update({'mean': new_mean, 'count': new_cnt})
        child: Union[Self, IndexLeaf] = self.get_child_at(selected_leaf_idx[0])
        child.update(algorithm, selected_leaf_idx[1:], score, kth_largest_score)

    def update_epsgreedy(self, algorithm: str, selected_leaf_idx: List[int], score: float, kth_largest_score: float):
        self.metadata['histogram'].update_from_score(score, kth_largest_score)
        child: Union[Self, IndexLeaf] = self.get_child_at(selected_leaf_idx[0])
        child.update(algorithm, selected_leaf_idx[1:], score, kth_largest_score)


def store_index_to_json(index: IndexNode, filename: str):
    """
    Store the index_builder to a JSON file.

    :param index: The index_builder to be stored.
    :param filename: The name of the JSON file.
    """
    index_dict = index.to_dict()
    with open (filename, 'w') as json_file:
        json.dump(index_dict, json_file)

def load_index_from_json(filename: str) -> IndexNode:
    """
    Load the index_builder from a JSON file. TODO: implementation.

    :param filename: The name of the JSON file.
    :return: The index_builder loaded.
    """
    with open (filename, 'r') as json_file:
        return json.load(json_file)

def  get_index_from_dict(dict_: Dict, shuffle_elements: bool) -> Union[IndexNode, IndexLeaf]:
    """
    Get an index_builder from a dictionary representation.

    :param shuffle_elements: Whether to store the elements in leaf nodes in a shuffled manner or not.
    :param dict_: The dictionary representation of the index_builder.
    :return: The index_builder created from the dictionary.
    """
    if isinstance(dict_['children'][0], dict):
        children = [get_index_from_dict(child, shuffle_elements) for child in dict_['children']]
    else:
        return IndexLeaf(dict_['children'], metadata = None, shuffle = shuffle_elements)
    return IndexNode(children)
