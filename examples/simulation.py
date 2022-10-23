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
from linear_sscr import LinearCR

n = 100
m = 150
#n = 10
#m = 10
p = 2
d = 1


n_repititions = 6

# plt.figure(figsize=(7, 5))
plt.figure(figsize=(18, 15))

for ii in range(n_repititions):

	zx = onp.random.normal(size=(n, d))
	zy = onp.random.normal(size=(m, d))
	t = onp.random.normal(size=(n, d))
	W = onp.random.normal(size=(d, p))
	W = onp.array([[1,1]])
	S = onp.random.normal(size=(d, p))
	S = onp.array([[-1,1]])
	beta = onp.random.normal(size=(d, 1))
	beta = onp.array([[1]])
	sigma2 = 1e-2
	tau2 = 1e-2

	X = zx @ S + t @ W + onp.random.normal(scale=sigma2, size=(n, p))
	Y = zy @ S + onp.random.normal(scale=sigma2, size=(m, p))
	R = t @ beta + onp.random.normal(scale=tau2, size=(n, 1))

	model = LinearSSCR()
	model.fit(X, Y, R, d)

	preds = model.predict(X)


	plt.subplot(3, 3, ii + 1)
	plt.title("Repitition {}".format(ii + 1))
	plt.xlabel("R true")
	plt.scatter(R, preds)
	plt.ylabel("R_preds")
	# import ipdb; ipdb.set_trace()


plt.tight_layout()
plt.savefig("./out/simulation_example.png")
plt.show()


#import ipdb

#ipdb.set_trace()
