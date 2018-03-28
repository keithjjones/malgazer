import argparse
import random
import math
import time


# Original Window Entropy Function
def original_window_entropy():
    # Start the time counter
    starttime = time.process_time()

    # Entropy list
    H = [0 for i in range(1, k-j+2)]
    for x in range(1, k - j + 2):
        m = [0 for i in range(0, 256)]
        for y in range(1, j + 1):
            m[malware[x + y - 2]] += 1
        entropy = float(0)
        for y in range(0, 256):
            if m[y] != 0:
                entropy += -((float(m[y]) / j) * math.log2(float(m[y]) / j))
        H[x - 1] = entropy / K

    # End the time counter
    endtime = time.process_time()

    return endtime-starttime, H


# Optimized Window Entropy Function
def optimized_window_entropy():
    # Start the time counter
    starttime = time.process_time()

    # Entropy list
    H = [0 for i in range(1, k - j + 2)]

    # Create the count list
    m = [0 for i in range(0, 256)]

    # Initial population of count list
    for x in range(1, j + 1):
        m[malware[x - 1]] += 1

    # Compute initial entropy
    entropy = float(0)
    for y in range(0, 256):
        if m[y] != 0:
            entropy += -(float(m[y] / j) * math.log2(float(m[y] / j)))
    H[0] = entropy / K

    for x in range(2, k - j + 2):
        lastval = malware[x - 2]
        nextval = malware[x + j - 2]

        entropy += float(m[lastval] / j) * math.log2(float(m[lastval] / j))
        m[lastval] -= 1
        if m[lastval] != 0:
            entropy += -(
            float(m[lastval] / j) * math.log2(float(m[lastval] / j)))

        if m[nextval] != 0:
            entropy += float(m[nextval] / j) * math.log2(float(m[nextval] / j))
        m[nextval] += 1
        entropy += -(float(m[nextval] / j) * math.log2(float(m[nextval] / j)))
        H[x - 1] = entropy / K

    # End the time counter
    endtime = time.process_time()

    return endtime - starttime, H

# Main functionality starts here...

# Argument parsing
parser = argparse.ArgumentParser(description='Calculates the entropy of a file.')
parser.add_argument('Size',
                    help='The size, in bytes, to simulate a malware file.',
                    type=int)
parser.add_argument("-w", "--window",
                    help="Window size, in bytes, for running entropy."
                         "", type=int, required=True)
parser.add_argument("-s", "--seed",
                    help="The seed value to the random function to simulate a malware file."
                         "", default=0, type=int, required=False)
parser.add_argument("-K", "--K",
                    help="The K scale value. Must not be zero."
                         "", default=1, type=int, required=False)
parser.add_argument("-a", "--average",
                    help="The number of runs for averaging, greater than zero."
                         "", default=1, type=int, required=False)

args = parser.parse_args()

k = args.Size
j = args.window
K = args.K

# Create a random file, based upon seed, for the simulated malware file.
random.seed(args.seed)
# Random Data
malware = [random.randint(0, 255) for i in range(1, k+1)]
# No Entropy Data
# malware = [1 for i in range(0, 256)]
# Full Entropy Data
# malware = [i for i in range(0, 256)]
# print("Malware: {0}".format(malware))

# Create running time counters...
totalorigtime = 0
totalopttime = 0

for a in range(0, args.average):
    # Calculate the original window entropy algorithm
    origtimespan, origH = original_window_entropy()
    totalorigtime += origtimespan

    # Calculate the optimized window entropy algorithm
    opttimespan, optH = optimized_window_entropy()
    totalopttime += opttimespan

    # Error checking to check for algorithm similarities
    if len(origH) != len(optH):
        print("ERROR: H's not same length!")
        print("Original H Length: {0}".format(len(origH)))
        print("Optimized H Length: {0}".format(len(optH)))
        exit(-1)
    #
    # if origH != optH:
    #     print("ERROR: H's not the same!")
    #     print("Original H: {0}".format(origH))
    #     print("Optimized H: {0}".format(optH))
    #     print("Difference: {0}".format([origH[i] - optH[i] for i in range(0, len(optH))]))
    #     exit(-1)

# print("Entropy: {0}".format(H))
print("Original Algorithm Average Running Time for {1:,} Iterations: {0:.4E}".format(totalorigtime/args.average, args.average))
print("Optimized Algorithm Average Running Time for {1:,} Iterations: {0:.4E}".format(totalopttime/args.average, args.average))

# print("Original algorithm entropy vector: {0}".format(origH))
# print("Optimized algorithm entropy vector: {0}".format(optH))
# print("Difference: {0}".format([origH[i] - optH[i] for i in range(0, len(optH))]))
