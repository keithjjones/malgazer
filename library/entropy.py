# Entropy Module
import math
from collections import Counter
from collections import deque


class RunningEntropy(object):
    """Class to hold running entropy calculations."""
    def __init__(self, window=256, normalize=True):
        """
        Class to hold running entropy calculations.

        :param window:  The size, in bytes, of the window to calculate for
            running entropy.  Must be larger than one.
        :param normalize:  If True, will normalize the entropy between 0 and 1.
        :return:  Nothing.
        """
        if window < 2:
            raise ValueError

        # Default for our data is None
        self.entropy_data = None

        self.window = window
        self.normalize = normalize

    def calculate(self, inputbytes):
        """
        Calculates the running entropy of inputbytes with the window size.
        This algorithm is optimized for single thread speed.

        :param inputbytes: A list of values representing bytes.
        :return: A list of entropy values for the running window, None on error.
        """
        if len(inputbytes) < self.window:
            return None

        # The counts
        bytescounter = Counter()
        # The current window for calculations
        byteswindow = deque()
        # The data to calculate
        inputbytesqueue = deque(inputbytes)

        entropy_list = list()

        # Initially fill up the window
        for i in range(0, self.window):
            currchar = inputbytesqueue.popleft()
            byteswindow.append(currchar)
            bytescounter[currchar] += 1

        # Calculate the initial entropy
        current_entropy = float(0)
        for b in bytescounter:
            if bytescounter[b] > 0:
                current_entropy -= ((float(bytescounter[b]) / self.window) *
                                    math.log2(float(bytescounter[b]) /
                                              self.window))

        # Add entropy value to output
        entropy_list.append(current_entropy)

        while True:
            # If there is no more input, break
            try:
                currchar = inputbytesqueue.popleft()
            except IndexError:
                break

            # Remove the old byte from current window and calculate
            oldchar = byteswindow.popleft()
            current_entropy += ((float(bytescounter[oldchar]) / self.window) *
                               math.log2(float(bytescounter[oldchar]) /
                                         self.window))
            bytescounter[oldchar] -= 1
            if bytescounter[oldchar] > 0:
                current_entropy -= ((float(bytescounter[oldchar]) / self.window)
                                    * math.log2(float(bytescounter[oldchar]) /
                                             self.window))

            # Calculate the newest added byte to the window
            if bytescounter[currchar] > 0:
                current_entropy += ((float(bytescounter[currchar]) /
                                     self.window) *
                                    math.log2(float(bytescounter[currchar]) /
                                              self.window))
            byteswindow.append(currchar)
            bytescounter[currchar] += 1
            current_entropy -= ((float(bytescounter[currchar]) / self.window) *
                                math.log2(float(bytescounter[currchar]) /
                                          self.window))

            # Add entropy value to output
            entropy_list.append(current_entropy)

        # Normalize if desired
        if self.normalize is True:
            self.entropy_data = [i/8 for i in entropy_list]
        else:
            self.entropy_data = entropy_list

        # Return the data
        return self.entropy_data
