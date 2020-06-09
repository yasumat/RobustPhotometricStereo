import numpy as np
import warnings


def L1_residual_min(A, b, max_ite=1000, tol=1.0e-8):
    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)
        minimize ||Ax - b||_1

    :param A: A design matrix (numpy 2D array)
    :param b: A column vector as a numpy 2D array
    :param max_ite:Maximum number of iterations
    :param tol: Tolerance
    :return: An approximate solution `x` that minimizes ||Ax - b||_0.

    Raises:
        ValueError: An error occurs in evaluating the dimensionality of the input matrix A and vector b.
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    eps = 1.0e-8
    m, n = A.shape

    xold = np.ones((n, 1))
    W = np.identity(m)
    if np.ndim(b) != 2 and b.shape[1] != 1:
        raise ValueError("b needs to be a column vector m x 1")

    iter = 0
    while iter < max_ite:
        iter = iter + 1
        # Solve the weighted least squares WAx=Wb
        x = np.linalg.lstsq(W.dot(A), W.dot(b), rcond=None)[0]
        r = b - A.dot(x)
        # Termination criterion
        if np.linalg.norm(x - xold) < tol:
            return x
        else:
            xold = x
        # Update weighting factor
        W = np.diag(np.asarray(1.0 / np.maximum(np.sqrt(np.fabs(r)), eps))[:, 0])
    return x


def sparse_bayesian_learning(A, b, max_ite=1000, tol=1.0e-8):
    """Derives approximate solution to minimize ||Ax - b||_0.

    Computes L0 residual minimization by sparse bayesian learning
    to derive an approximate solution for minimize ||Ax - b||_0.

    :param A: A design matrix (numpy 2D array)
    :param b: A column vector as a numpy 2D array
    :param MAX_ITER:Maximum number of iterations
    :param tol: Tolerance
    :return: An approximate solution `x` that minimizes ||Ax - b||_0.

    Raises:
        ValueError: An error occurs in evaluating the dimensionality of the input matrix A and vector b.

    """
    GAMMA_THR = 1e-8    # For numerical stability
    lambda1 = 1.0    # Coefficient regularizer
    lambda2 = 1.0e-6

    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    m, n = A.shape

    gamma = np.ones((m, 1))
    x_old = 1000 * np.ones((n, 1))
    ite = 0
    while ite < max_ite:
        W = np.diag((1. / gamma.T)[0])
        C = lambda2 * np.identity(n) + A.T.dot(W).dot(A)
        d = A.T.dot(W).dot(b)
        x = np.linalg.solve(C, d)
        if np.linalg.norm(x - x_old) < tol:
            break
        x_old = x
        e = b - A.dot(x)
        E = lambda1 * np.identity(n) + A.T.dot(W).dot(A)
        Xi = np.linalg.solve(E, A.T)
        Sigma_e_diag = np.sum(A * Xi.T, axis=1)
        gamma = e * e + np.array([Sigma_e_diag]).T
        gamma = np.maximum(gamma, GAMMA_THR)
        ite = ite + 1
    return x


def pos(a):
    """
    Returns positive elements of the numpy array.
    Turns negative elements into zeros.
    """
    return a * np.double(a > 0)


def neg(a):
    """
    Returns negative elements of the numpy array.
    Turns positive elements into zeros.
    """
    return a * np.double(a < 0)


def shrinkage(x, kappa):
    """
    Shrinkage operation
    """
    return pos(x - kappa) - pos(-x - kappa)


def rpca_inexact_alm(D, lambda_=None, max_ite=1000, tol=1.0e-6):
    """Computes Robust-PCA by inexact ALM.

    Computes Robust-PCA matrix decomposition D = A + E by the inexact augmented Lagrangian multiplier method.
    ``The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices,''
    Z. Lin, M. Chen, L. Wu, and Y. Ma (UIUC Technical Report UILU-ENG-09-2215, November 2009).
    The decomposition is performed by solving the following problem:
        minimize_{A, E} ||A||_* + lambda_ * ||E||_1 s.t. D = A + E

    :param D: Input matrix to be decomposed
    :param lambda_: A weighting parameter for controlling the low-rankness and sparsity
    :param max_ite: Maximum number of iterations
    :param tol: Tolerance
    :return:
        A: Low-rank matrix
        E: Sparse error matrix
        ite: Number of iterations
    Raises:
        ValueError: An error occurred in evaluating the dimensionality of the input matrix D.

    """
    if D.ndim != 2:
        raise ValueError("Input matrix D needs to be a matrix.")

    (m, n) = D.shape
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(m, n))

    Y = D
    norm_two = np.linalg.svd(Y, full_matrices=False, compute_uv=False)[0]
    norm_inf = np.max(np.abs(Y[:])) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    A = np.zeros((m, n))
    E = np.zeros((m, n))
    mu = 1.25 / norm_two   # This can be tuned
    mu_bar = mu * 1e7
    rho = 1.5    # This can be tuned
    d_norm = np.linalg.norm(D, 'fro')
    ite = 0
    sv = 10    # This can be tuned

    while ite < max_ite:
        ite = ite + 1
        T = D - A + (1/mu) * Y
        E = np.maximum(T - lambda_/mu, 0.0)
        E = E + np.minimum(T + lambda_ / mu, 0.0)
        U, S, V = np.linalg.svd(D - E + (1.0 / mu) * Y, full_matrices=False)
        svp = len(S[S > 1.0 / mu])
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)
        A = U[:, 0:svp].dot(np.diag(S[0:svp] - 1.0/mu)).dot((V[0:svp, :]))
        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)
        stop_criterion = np.linalg.norm(Z, 'fro') / d_norm
        if stop_criterion < tol:
            return A, E, ite

    warnings.warn("Exceeded the maximum number of iterations.")
    return A, E, ite
