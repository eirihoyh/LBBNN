from __future__ import annotations
import numpy as np

def get_data(n=10_000, beta=np.array([1,5]), classification=True, non_lin=False):

    # Generate some feature data
    np.random.seed(42)
    X = np.random.randn(n,1)  # 100 samples, 2 features
    # Add intercept term
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

    if non_lin:
        # Non-linear relationship
        y = beta[0] + beta[1]*(X**2).flatten()
    else:
        # Linear relationship
        y = beta[0] + beta[1]*X.flatten()
    # # Compute linear predictor
    # y = X_with_intercept @ beta

    if classification:
        # Apply custom logic function to get probabilities
        p = 1/(1+np.exp(-y))
        # Sample from Bernoulli distribution
        y = np.random.binomial(1, p)


    return X, y, X_with_intercept


def create_data_unif(n: int, beta=(10,1,1,1,1), dep_level: float = 0.5, classification: bool = False, non_lin: bool = False, seed: int | None = None):
    rng = np.random.default_rng(seed)
    x0 = np.ones(n)
    x1 = rng.uniform(-10,10,n); x2 = rng.uniform(-10,10,n); x3 = rng.uniform(-10,10,n); x4 = rng.uniform(-10,10,n)
    x3 = dep_level * x1 + (1 - dep_level) * x3
    y = beta[0] + beta[1] * x1 + beta[2] * x2 if not non_lin else beta[0] + beta[1]*x1 + beta[2]*x2 + beta[3]*x1**2 + beta[4]*x2**2 + x1*x2
    y = y + rng.normal(scale=0.01, size=n)
    if classification:
        y = y - y.min(); y = y / max(y.max(), 1e-12); y = (y > np.median(y)).astype(int)
    return y, np.column_stack((x0,x1,x2,x3,x4))

def create_bsr_data(n: int, urange=(-3,3), func: int = 1, seed: int | None = None):
    rng = np.random.default_rng(seed)
    l,u = urange[0], urange[1]
    x1 = np.linspace(-1.5,1.5,n); x2 = rng.uniform(l,u,n)
    if func == 1: y = 2.5*x1**4 - 1.3*x1**3 + 0.5*x2**2 - 1.7*x2
    elif func == 2: y = 8*x1**2 + 8*x2**3 - 15
    elif func == 3: y = 0.2*x1**3 - 0.5*x1 + 0.5*x2**3 - 1.2*x2
    elif func == 4: y = 1.5*np.exp(x1) + 5*np.cos(x2)
    elif func == 5: y = 6*np.sin(x1)*np.cos(x2)
    elif func == 6: y = 1.35*x1*x2 + 5.5*np.sin((x1-1)*(x2-1))
    elif func == 7:
        rand0 = rng.normal(scale=0.02, size=n); y = x1 + 0.3*np.sin(2*np.pi*(x1+rand0)) + 0.3*np.sin(4*np.pi*(x1+rand0)) + rand0
    elif func == 8:
        rand0 = rng.normal(scale=1.0, size=n); y = 10*np.sin(2*np.pi*x1) + rand0
    else: raise ValueError('func must be in {1,2,3,4,5,6,7,8}')
    return y, np.column_stack((x1,x2))
