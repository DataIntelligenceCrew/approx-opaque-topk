from typing import List, Dict
from typing_extensions import Self


class Histogram:
    """
    Abstracts the maintenance of a histogram.
    Histograms are used to approximate distributions and compute expected marginal gain.
    """

    def __init__(self, bin_borders: List[float], rebin_decay: float = 0.95, enlarge_max_factor: float = 1.2, enlarge_lowest: bool = True):
        """
        Initialize a Histogram instance.

        :param bin_borders: An increasing list of bin borders.
                            There must be two or more bins and the borders must be non-negative.
        :param rebin_decay: The factor to decay previous bin counts when re-binning.
        :param enlarge_max_factor: The factor to enlarge the bin borders when a sample larger than the current maximum
                               range is observed.
        :param enlarge_lowest: Whether to enlarge the lowest bin when the kth largest score is larger than the top-end
                               of the 2nd lowest bin.
        """
        # Save parameters
        self._num_bins: int = len(bin_borders) - 1
        self._bin_borders: List[float] = bin_borders
        self._bin_counts: List[float] = [0.0 for _ in range(self._num_bins)]  # Initialize a count of 0 for each bin
        self._total_counts: float = 0.0 # The sum of all counts in this histogram; initialize > 0 to avoid division by 0
        self._rebin_decay: float = rebin_decay
        self._enlarge_factor: float = enlarge_max_factor
        self._enlarge_lowest: bool = enlarge_lowest

    @staticmethod
    def new_empty_uniform(min_value: float, max_value: float, num_bins: int, rebin_decay: float = 0.95, enlarge_max_factor: float = 1.2, enlarge_lowest: bool = True) -> 'Histogram':
        """
        Creates a new empty histogram with uniform bins.

        :param min_value: The minimum value of the interval.
        :param max_value: The maximum value of the interval.
        :param num_bins: The number of bins in the interval. Must be at least 1.
        :param rebin_decay: The factor to decay previous bin counts when re-binning.
        :param enlarge_max_factor: The factor to enlarge the bin borders when a sample larger than the current maximum
                               range is observed.
        :param enlarge_lowest: Whether to enlarge the lowest bin when the kth largest score is larger than the top-end
                               of the 2nd lowest bin.
        :return: A new Histogram instance with uniform bins.
        """
        bin_borders: List[float] = uniformly_divide_range(min_value, max_value, num_bins)
        return Histogram(bin_borders, rebin_decay, enlarge_max_factor, enlarge_lowest)

    def expected_marginal_gain(self, kth_largest_score: float):
        """
        Computes the expected marginal gain in S_(k) when adding a new sample randomly drawn from the distribution
        modelled by this histogram. (Uses uniform value assumption, i.e. a piecewise-continuous distribution.)

        :param kth_largest_score: The current S_(k) value.
        :return: The expected marginal gain in S_(k) when adding a new sample from this histogram's distribution.
        """
        # Iterate over each bin to compute expected marginal gain
        gain = 0.0
        for b in range(self._num_bins):
            lo: float = max(kth_largest_score, self._bin_borders[b])
            hi: float = self._bin_borders[b + 1]
            if hi > lo:  # If S_(k) is above this bin, we can skip this bin
                bin_mean: float = (hi + lo) / 2.0
                bin_prob: float = (self._bin_counts[b] / self._total_counts) * (hi - lo) / (hi - self._bin_borders[b]) if self._total_counts > 0.0 else 0.0
                bin_gain: float = bin_prob * bin_mean
                gain += bin_gain
        #print(self, "gain:", gain, "Rk:", kth_largest_score)
        return gain

    def rebin(self, new_bin_borders: List[float]):
        """
        Updates this histogram's bins with new bin borders using the uniform value assumption.

        :param new_bin_borders: Bin borders for the new histogram.
        """
        # Compute the new bin counts by comparing the old and new bins
        new_bin_counts: List[float] = [0.0 for _ in range(len(new_bin_borders)-1)]
        for b_old in range(self._num_bins):
            for b_new in range(self._num_bins):
                # Compute the overlap between the two bins
                lo: float = max(self._bin_borders[b_old], new_bin_borders[b_new])
                hi: float = min(self._bin_borders[b_old+1], new_bin_borders[b_new+1])
                if hi > lo:
                    cnt: float = self._bin_counts[b_old] * (hi - lo) / (self._bin_borders[b_old+1] - self._bin_borders[b_old])
                    new_bin_counts[b_new] += cnt
        # Decay the new bin counts
        new_bin_counts: List[float] = [self._rebin_decay * x for x in new_bin_counts]
        new_total_counts: float = sum(new_bin_counts)
        # Replace the counts in new histogram
        self._bin_counts = new_bin_counts
        self._total_counts = new_total_counts
        self._bin_borders = new_bin_borders

    def subtract(self, other: Self):
        for b_self in range(self._num_bins):
            for b_new in range(self._num_bins):
                # Compute the overlap between the two bins
                lo: float = max(self._bin_borders[b_self], other._bin_borders[b_new])
                hi: float = min(self._bin_borders[b_self+1], other._bin_borders[b_new+1])
                if hi > lo:
                    cnt: float = self._bin_counts[b_self] * (hi - lo) / (self._bin_borders[b_self+1] - self._bin_borders[b_self])
                    self._bin_counts[b_self] = max(self._bin_counts[b_self] - cnt, 0.0)

    def update_from_score(self, score: float, kth_largest_score: float):
        """
        Update this histogram based on a single sample of a score. If the score falls into the histogram's range, then
        simply increases the appropriate bin's count by 1. If the score is larger than the histogram's range, then
        re-bins the histogram to include the new score.

        :param score: The new score to update the histogram with.
        :param kth_largest_score: The current S_(k) value. Potentially used to enlarge the size of the lowest bin.
        """
        #print("updating from score histogram!")
        # Rebin if the score is larger than maximum range
        range_max: float = self._bin_borders[-1]
        if max(score, kth_largest_score) > range_max:
            #print("rebin due to large score of", score, kth_largest_score, "compared to previous max", range_max)
            new_range_max: float = max(score, kth_largest_score) * self._enlarge_factor
            new_borders_tail: List[float] = uniformly_divide_range(self._bin_borders[1], new_range_max, self._num_bins-1)
            new_borders: List[float] = [self._bin_borders[0]] + new_borders_tail
            #print("new borders:", new_borders)
            self.rebin(new_borders)
            #print(self)
        # Rebin if enlarge_lowest flag is set to True and the kth largest score is greater than the top-end of the 2nd lowest bin
        if self._enlarge_lowest:
            if kth_largest_score > self._bin_borders[2]:
                #print("enlarge lowest as S_(k) is", kth_largest_score, "compared to bin border", self._bin_borders[2])
                new_max_range: float = kth_largest_score * self._enlarge_factor if kth_largest_score >= self._bin_borders[-1] else self._bin_borders[-1]
                new_borders_tail: List[float] = uniformly_divide_range(kth_largest_score, new_max_range, self._num_bins - 1)
                new_borders: List[float] = [self._bin_borders[0]] + new_borders_tail
                #print("new borders:", new_borders)
                self.rebin(new_borders)
                #print(self)
        # Check each bin to see if the score falls into it, and increment the count if so
        for b in range(self._num_bins):
            b_hi: float = self._bin_borders[b+1]
            if score <= b_hi:  # Score falls into this bin
                self._bin_counts[b] += 1.0
                break
        # Increment the total count
        self._total_counts += 1.0
        #print(self)
        #print()

    def to_dict(self) -> Dict:
        """
        :return: A dictionary representation of this histogram. Useful for JSON serialization.
        """
        return {
            "num_bins": self._num_bins,
            "bin_borders": self._bin_borders,
            "bin_counts": self._bin_counts,
            "total_counts": self._total_counts
        }

    def __repr__(self):
        return str(self.to_dict())

    def get_count(self) -> float:
        return self._total_counts

def uniformly_divide_range(min_value: float, max_value: float, num_bins: int) -> List[float]:
    """
    Given an interval of two numbers, uniformly divides the interval into a specified number of bins.
    The bins are specified by an increasing sequence of (num_bins+1) numbers [min_value, ...... , max_value].

    Examples:
      - interval = [0, 1], num_bins = 2 -> [0.0, 0.5, 1.0]
      - interval = [0, 1], num_bins = 1 -> [0.0, 1.0]

    :param min_value: The minimum value of the interval.
    :param max_value: The maximum value of the interval.
    :param num_bins: The number of bins in the interval. Must be at least 1.
    :return: An increasing list of bin borders.
    """
    # Division computation
    return [min_value + (max_value - min_value) * (x / num_bins) for x in range(num_bins + 1)]
