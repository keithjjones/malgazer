# Algorithm from https://developer.ibm.com/streamsdev/docs/anomaly-detection-in-streams/ written in pure Python
from collections import deque


class AnomalyDetector(object):
    """Anomaly Detector class."""
    def __init__(self, data, ref_size, pattern_size):
        """
        Creates an Anomaly Detector object.

        :param data: A list of data points
        :param ref_size: The size of the reference window for detection.
        :param pattern_size: The size of the pattern window for detection
        """
        self.data = data
        self.pattern_size(pattern_size)
        self.reference_size(ref_size)

    def reference_size(self, ref_size=None):
        """
        Gets/Changes the reference window size.
        :param ref_size:
        :return:
        """
        if ref_size is None:
            return self._ref_size
        if ref_size >= self.pattern_size():
            raise ValueError("Reference window size {0} needs to be smaller "
                             "than pattern window size!".format(ref_size))
        if ref_size < 1:
            raise ValueError("Reference window size {0} needs to be a "
                             "positive number!".format(ref_size))
        self._ref_size = ref_size
        return self._ref_size

    def pattern_size(self, pattern_size=None):
        if pattern_size is None:
            return self._pattern_size
        if pattern_size < 1:
            raise ValueError("Pattern window size {0} needs to be a "
                             "positive number!".format(pattern_size))

        if pattern_size >= len(self.data):
            raise ValueError("Pattern window size {0} needs to be less "
                             "than the data size of {1}!".format(pattern_size,
                                                                 len(self.data)))
        self._pattern_size = pattern_size
        return self._pattern_size

    def calculate(self):
        """
        Calculate the anomaly differences.
        :return: A list of differences from anomaly detection.
        """
        # This is the larger pattern window
        pattern_window = deque()
        # This is the smaller reference window
        reference_window = deque()

        # This holds the indices with anomalies
        self.anomaly_diffs = list()

        # Run this on every data point
        for i in range(0, len(self.data)):
            # Fill up our larger pattern window...
            if len(pattern_window) < self.pattern_size():
                pattern_window.append(self.data[i])
                # Fill up our reference window...
                if i >= self.pattern_size() - self.reference_size():
                    reference_window.append(self.data[i])
                continue

            newdata = self.data[i]

            # Update the reference window
            reference_window.popleft()
            reference_window.append(newdata)

            diff = 0
            # Calculate the sliding window differences...
            for j in range(0, self.pattern_size() - self.reference_size() + 1):
                # Calculate the differences...
                for k in range(0, self.reference_size()):
                    diff += abs(pattern_window[j+k] - reference_window[k])

            self.anomaly_diffs.append((i, diff))

            # Update the pattern window
            pattern_window.popleft()
            pattern_window.append(newdata)

        return self.anomaly_diffs
