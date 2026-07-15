from mlp import MLP
from value import Value

n = MLP(3, [4, 4, 1])

x = [[2.0, 3.0, 1.0],
     [-1.4, 1.3, 1.6],
     [-1.3, -2.9, -0.3],
     [-1.6, 1.3, -2.4]]

y = [-1.0, 1.0, -1.0, 1.0]


for k in range(50000):
    # forward pass
    ypred = [n(xi) for xi in x]
    loss = sum(((yp + (-1 * yi)) * (yp + (-1 * yi)) for yp, yi in zip(ypred, y)), Value(0))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update step
    for parameter in n.parameters():
        parameter.val += -0.01 * (parameter.grad)
    
    if k % 100 == 0:
        print(k, loss.val)


res = n([2.0, 3.0, 1.0])
print(res)
