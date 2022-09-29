import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

import sys

sys.path.append("../models")
from linear_sscr import LinearSSCR

n = 100
m = 150
p = 2
d = 1

n_repititions = 9

# plt.figure(figsize=(7, 5))
plt.figure(figsize=(18, 15))

for ii in range(n_repititions):

	zx = onp.random.normal(size=(n, d))
	zy = onp.random.normal(size=(m, d))
	t = onp.random.normal(size=(n, d))
	W = onp.random.normal(size=(d, p))
	S = onp.random.normal(size=(d, p))
	beta = onp.random.normal(size=(d, 1))
	sigma = 1e-2
	tau = 1e-2

	X = zx @ S + t @ W + onp.random.normal(scale=sigma)
	Y = zy @ S + onp.random.normal(scale=sigma)
	R = t @ beta + onp.random.normal(scale=tau)


	model = LinearSSCR()
	model.fit(X, Y, R, d)


	preds = model.predict(X)

	plt.subplot(3, 3, ii + 1)
	plt.title("Repitition {}".format(ii + 1))
	plt.scatter(R, preds)
	plt.xlabel("R true")
	plt.ylabel("R estimated")

plt.tight_layout()
plt.savefig("./out/simulation_example.png")
plt.show()
import ipdb

ipdb.set_trace()
