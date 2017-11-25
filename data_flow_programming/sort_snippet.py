# encoding=utf-8
import numpy as np

X = np.random.rand(20)
# insection sort
x = np.copy(X)
for i in range(len(x)):
    for j in range(i, 0, -1):
        if x[j - 1] > x[j]:
            t = x[j - 1]
            x[j - 1] = x[j]
            x[j] = t
        else:
            break
print(x)

# bubble sort
x = np.copy(X)
for i in range(len(x)):
    for j in range(len(x) - 1, i, -1):
        if x[j - 1] > x[j]:
            t = x[j - 1]
            x[j - 1] = x[j]
            x[j] = t
print(x)
