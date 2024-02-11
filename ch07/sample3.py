import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
import dezero.functions as F

# データの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

# プロット
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue')
plt.title('Randomly Generated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x,W) + b
    return y

def mse(x0,x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mse(y,y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print("===")
print("W = ",W.data)
print("b = ",b.data)

# # Plot
# # plt.scatter(x.data, y.data, s=10)
# plt.scatter(x.data.flatten(), y.data.flatten(), s=10)

# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = predict(t)
# plt.plot(t, y_pred.data, color='r')
# plt.show()

# x.data と y.data が memoryview オブジェクトである場合
x_array = np.asarray(x.data)
y_array = np.asarray(y.data)

# NumPy 配列を1次元配列に変換
x_flattened = x_array.flatten()
y_flattened = y_array.flatten()

# 1次元配列を scatter 関数に渡す
plt.scatter(x_flattened, y_flattened, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
