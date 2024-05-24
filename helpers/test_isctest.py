import numpy as np
import sys
sys.path.append('../')
from tracking import isctest

def add_group(groups, t1, t2):
    added = False
    for group in groups:
        if t1 in group or t2 in group:
            group.add(t1)
            group.add(t2)
            added = True
            break
    if not added:
        groups.append({t1, t2})

components = np.load("../isctest/isctest_2.npy")
K = 16

converted = []
grouped = []
clusters = isctest(components)
norm_c = np.ceil(clusters/(K+1))

#print(norm_c)

for cluster in norm_c:
    new_cluster = []
    for i, element in enumerate(cluster):
        element_int = int(element)
        if element_int > 0:
            if i < 6:
                item = ('finger', i + 1, element_int)
            else:
                item = ('fist', i + 1, element_int)
            new_cluster.append(item)
    converted.append(new_cluster)

#print(converted)

for cluster in converted:
    for i in range(len(cluster) - 1):
        add_group(grouped, cluster[i], cluster[i+1])

#print(grouped)

sorted_group = []
for g in grouped:
    sorted_group.append(sorted(list(g)))

for sg in sorted_group:
    print(sg)