#!/usr/bin/python
"""visualize performance for different versions"""

import matplotlib.pyplot as plt
import csv
import numpy as np

tsvfiles = ["cuda.stats", "thrust_prefix.stats", "thrust_foreach.stats","sequential.stats"]
labels = ["Pure CUDA", "Thrust with prefix scan", "Thrust with foreach", "Sequential 2"]
color = ['b', 'g', 'r', 'c']

X = []
for tsvfile in tsvfiles[:4]:
    data = []
    with open(tsvfile, 'rb') as csvfile:
        tsvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        tmp = []
        for row in tsvreader:
            tmp.append(row)
            i+=1
            if i%20 == 0:
                data.append(np.mean(np.array(tmp, dtype=float), axis=0))
                tmp = []
            
    X.append(np.array(data))

plt.figure()
plt.title("Performance of horizontal convolution (incl. data transfer)")
plt.xlabel("N for NxN images")
plt.ylabel("Time in seconds")


# sequential
x = X[0][:,0]
y = X[0][:,2]
plt.plot(x, y,'k', label="Original sequential")

# parallel
for i in range(2):
    x = X[i][:,0]
    y = X[i][:,1]
    plt.plot(x, y, color[i], label=labels[i])

i=2
# foreach
x = X[i][:,0]
y = X[i][:,1] + X[i][:,3]/2.0 + X[i][:,4]
plt.plot(x, y, color[i], label=labels[i])


i=3
# seq 2
x = X[i][:,0]
y = X[i][:,1] + X[i][:,3]/2.0
plt.plot(x, y, color[i], label=labels[i])

plt.ylim([0,0.6])
plt.xlim([0,2000])
plt.legend(loc=0)
plt.savefig("performance.png")
