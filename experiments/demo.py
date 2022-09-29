import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

import sys

sys.path.append("../models")
from linear_sscr import LinearSSCR

n = 100
m = 150
#n = 10
#m = 10
p = 2
d = 1


zx = onp.random.normal(size=(n, d))
zy = onp.random.normal(size=(m, d))
t = onp.random.normal(size=(n, d))
W = onp.random.normal(size=(d, p))
W = onp.array([[1,1]])
S = onp.random.normal(size=(d, p))
S = onp.array([[-1,1]])
beta = onp.random.normal(size=(d, 1))
beta = onp.array([[1]])
sigma2 = 1e-1
tau2 = 1e-1

X = zx @ S + t @ W + onp.random.normal(scale=sigma2, size=(n, p))
Y = zy @ S + onp.random.normal(scale=sigma2, size=(m, p))
R = (t @ beta).squeeze() + onp.random.normal(scale=tau2, size=n)

import ipdb; ipdb.set_trace()

model = LinearSSCR()
model.fit(X, Y, R, d)

preds = model.predict(X)


plt.figure(figsize=(6, 6))
plt.scatter(Y[:, 0], Y[:, 1], c="black", label="Background")
plt.scatter(X[:, 0], X[:, 1], c=R, label="Foreground")
plt.colorbar()
plt.legend()
plt.savefig("./out/simulation_demo.png")
plt.show()
import ipdb; ipdb.set_trace()


