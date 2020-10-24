#!/usr/bin/env python3
import sys
import numpy as np
g = {}
uni = []
values = []
neighbors = []
reduce = sys.stdin

for line in reduce:
    split = line.split()
    values.append(int(split[0]))
    neighbors.append(int(split[1]))
    # print(split)
    uni.append((split[0]))
uni = np.sort(np.unique(uni))
# print(uni)

# print(values)
# print(neighbors)

i = 7
j = 0
pred_list = []
for x in uni:
    pred_list.append(str(max(set(neighbors[j:i]))))
    i += 7
    j += 7
# print(pred_list)

# print(label)
c = 0
for x in pred_list:
    print("The label of ", uni[c], " is ", x.replace("]", " "))
    c += 1
