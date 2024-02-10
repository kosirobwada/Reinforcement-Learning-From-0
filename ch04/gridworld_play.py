import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import numpy as np
from common.gridworld import GridWorld
from collections import defaultdict

env = GridWorld()
# V = defaultdict(lambda:0)
# for state in env.states():
#     V[state] = np.random.randn()
# state = (1,2)
# print(V[state])
# env.render_v(V)

pi = defaultdict(lambda: {0:0.25,1:0.25,2:0.25,3:0.25})

state = (0,1)
print(pi[state])