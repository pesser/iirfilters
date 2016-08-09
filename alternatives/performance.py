#!/usr/bin/python
"""visualize performance for different versions"""

import matplotlib.pyplot as plt
import csv

tsvfiles = ["cuda.stats", "thrust_prefix.stats"]
labels = ["Pure CUDA", "Thrust with prefix scan"]
color = ['blue', 'red']

X = []
for tsvfile in tsvfiles:
    data = {}
    with open(tsvfile, 'r') as csvfile:
        tsvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in tsvreader:
            data[int(row[0])] = [float(row[1]), float(row[2]), float(row[2])]
    X.append(data)


plt.figure()
plt.title("Performance of convolution")
plt.xlabel("N for NxN images")
plt.ylabel("Time in microseconds")


x = sorted(X[0].keys())

# sequential
y = [X[0][k][1] for k in x]
plt.plot(x, y, 'black', label="Sequential")

# parallel
for i in range(len(tsvfiles)):
    y = [X[i][k][0] for k in x]
    plt.plot(x, y, color[i], label=labels[i])


plt.legend(loc=0)
plt.savefig("performance.png")
