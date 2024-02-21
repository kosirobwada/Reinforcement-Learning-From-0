import numpy as np
from dezero import Model
import  dezero.layers as L
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi*x) + np.random.rand(100,1)

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y
    
model = TwoLayerNet(10,1)

for param in model.params():
    print(param)

model.cleargrads()

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    
    if i % 100 == 0:
        print(loss)