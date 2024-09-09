from min_max_heap import MinMaxHeap
from typing import List, Tuple, Dict


class LimitedPQ:
    """
    A max-priority-queue that ensures that the size of the queue is bounded by k.
    """
    def __init__(self, cardinality_constraint: int):
        self._k: int = cardinality_constraint
        self._heap: MinMaxHeap = MinMaxHeap()

    def insert(self, element: str, score: float):
        """
        Insert an element and its score into the queue.
        If the queue size exceeds k, the element with the lowest score is removed.
        :param element: The element to insert. It is usually the string identifier for an object.
        :param score: The score of the element.
        """
        # Since MinMaxHeap is a min-heap, but we want a max-heap, we negate the score to get the priority
        self._heap.push((-score, element))
        if self._heap.size() > self._k:  # Check if cardinality constraint is violated
            self._heap.pop_max()  # Remove "max" which is actually the element with the lowest score

    def get_heap(self) -> Tuple[List[str], List[float]]:
        """
        :return: A list of the elements in the queue and a list of their corresponding scores, in descending order of score.
        """
        heap: List = self._heap.queue()
        heap.sort()  # Sort the heap in ascending order of priority, which is descending order of score
        elements, scores = zip(*heap)  # Unzip the list of tuples into two lists
        scores: List[float] = [-score for score in scores]  # Negate the priorities to get the actual scores
        return elements, scores

    def kth_best_score(self) -> float:
        """
        :return: The score of the kth best element in the queue.
        """
        return -1 * self._heap.queue()[0][0]  # Get the first tuple in heap, then negate its priority

    def to_dict(self) -> Dict:
        """
        :return: A dictionary representation of the queue. Useful for JSON serialization.
        """
        elements, scores = self.get_heap()
        return {
            "elements": elements,
            "scores": scores,
            "kth_best_score": self.kth_best_score(),
            "sum_top_k": sum(scores)
        }
