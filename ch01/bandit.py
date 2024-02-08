import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms = 10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:    
            return 0

bandit = Bandit()

# for i in range(3):
#     print(bandit.play(0))

# Q = 0
# for n in range(1,11):
#     reward = bandit.play(0)
#     Q += (reward - Q) / n
#     print(Q)

# Qs = np.zeros(10)
# ns = np.zeros(10)
# for n in range(10):
#     action = np.random.randint(0,10)
#     reward = bandit.play(action)
#     ns[action] += 1
#     Qs[action] += (reward - Qs[action]) / ns[action]
#     print(Qs)

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action])/self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)