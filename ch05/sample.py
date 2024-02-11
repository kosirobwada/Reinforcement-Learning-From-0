if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch05.dice import sample
import numpy as np

trial = 100
samples = []
V, n = 0, 0

for _ in range(trial):
    s = sample()
    n += 1
    V += (s - V) / n
    samples.append(s)
    print(V)

# V = sum(samples)/len(samples)
# print(V)