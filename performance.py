"""visualize performance for different versions"""
#!/usr/bin/python
"""visualization of rbtree measurements"""

import matplotlib.pyplot as plt
import csv

tsvfiles = ["cuda.stats", "thrust.stats"]
labels = ["Pure CUDA", "Thrust with prefix scan"]


X = []
for tsvfile in tsvfiles:
    data = {}
    with open(tsvfile, 'rb') as csvfile:
        tsvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in tsvreader:
            data[int(row[0])] = [int(row[1]), int(row[2]), float(row[2])]
    X.append(data)


plt.figure()
plt.title("Performance of convolution")
plt.xlabel("N for NxN images")
plt.ylabel("Performance (ns)")


y = sorted(X[0].keys())
H = []

# sequential
x = [X[0][k][1] for k in y]

handle, = plt.plot(x, y, 'black', label="Sequential")
H.append(handle)

color = ['#999999', '#e41a1c', '#377eb8', '#4daf4a',
         '#984ea3', '#ff7f00', '#ffff33', '#a65628']
for i in range(len(tsvfiles)):
    x = [X[0][k][0] for k in y]
    handle, = plt.plot(x, y, color[i], label=labels[i])
    H.append(handle)


#plt.gca().set_ylim(0, 50000)
plt.legend(loc=2, bbox_to_anchor=(0.9, 1), prop={'size': 10})
plt.savefig("performance.png")
