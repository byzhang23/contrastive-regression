# Contrastive Linear Regression

The Python file for the linear contrastive regression model is `models/linear_cr_new.py`.

An example of how to fit the model and make predictions is below.

```python
n = 100  # Foreground sample size
m = 100  # Background sample size
p = 2    # Number of features
d = 1    # Latent dimension

zx = onp.random.normal(size=(n, d))    # Foreground shared latent variables
zy = onp.random.normal(size=(m, d))    # Background shared latent variables
t = onp.random.normal(size=(n, d))     # Foreground-specific latent variables
W = onp.random.normal(size=(d, p))     # Foreground-specific loadings matrix
S = onp.random.normal(size=(d, p))     # Shared loadings matrix
beta = onp.random.normal(size=(d, 1))  # Coefficient vector
sigma = 1e-2                           # Data matrix noise variance
tau = 1e-2                             # Response noise variance

# Foreground data
X = zx @ S + t @ W + onp.random.normal(scale=sigma, size=(n, 2))

# Background data
Y = zy @ S + onp.random.normal(scale=sigma, size=(m, 2))

# Response
R = t @ beta + onp.random.normal(scale=tau, size=(n, 1))

# Instantiate model
model = LinearContrastiveRegression()

# Fit model
model.fit(X, Y, R, d)

# Make predictions on training data
preds = model.predict(X)
```

## Citation

Zhang, B., Nyquist, S., Jones, A., Engelhardt, B. E., & Li, D. (2024). Contrastive linear regression. arXiv preprint arXiv:2401.03106. 