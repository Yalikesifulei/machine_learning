import numpy as np

def Normalize(X):
    """
    Normalize given data X (with features in columns) by subtracting mean
    and dividing by standard deviation.

    X (numpy.ndarray)   - data array.

    Returns normalized array and dictionary with mean and standard deviation of columns.
    """
    X_temp = X.copy()
    n = X_temp.shape[1]
    scales = {'mean' : np.zeros((1, n)), 'std' : np.zeros((1, n))}
    for j in range(n):
        scales['mean'][0, j] = np.mean(X_temp[:, j])
        scales['std'][0, j] = np.std(X_temp[:, j], ddof = 1)
        X_temp[:, j] -= scales['mean'][0, j]
        if scales['std'][0, j] >= 1e-6:
            X_temp[:, j] /= scales['std'][0, j]
    return X_temp, scales

def PCA(X, k = 0, tol = 0.99):
    """
    Perform principal component analysis (PCA).

    X (numpy.ndarray)   - data array.
    k (int)             - number of principal components. Default value zero used to choose k automatically.
    tol (float)         - percantage of variance retained (in range (0;1))

    Returns new data array and transition matrix.
    """
    m, n = X.shape
    if k > n:
        raise ValueError("Number of principal components must be less or equal to number of features.")
    if (tol <= 0) or (tol >= 1):
        raise ValueError("tol value must be in range (0;1).")
    Sigma = (1/m) * X.T @ X
    U, S = np.linalg.svd(Sigma)[:2]
    if k == 0:
        for i in range(n):
            if sum(S[:i]) >= tol*sum(S):
                k = i
                break
    Z = X @ U[:, 0:k]
    return Z, U[:, 0:k]