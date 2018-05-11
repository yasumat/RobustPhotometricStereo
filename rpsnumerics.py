import numpy as np
import warnings


def L1_residual_min(A, b, MAX_ITER=1000, tol=1.0e-8):
    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)
        minimize ||Ax - b||_1
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
    while iter < MAX_ITER:
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


def regularized_sparse_bayesian_learning(A, b, max_ite=1000, tol=1.0e-8):
    """Derives approximate solution to minimize ||Ax - b||_0.

    Computes L0 residual minimization by sparse bayesian learning with a regularizer
    to derive an approximate solution for minimize ||Ax - b||_0.

    Args:
        A: A design matrix (numpy 2D array)
        b: A column vector as a numpy 2D array
        max_ite: (optional) Maximum number of iterations
        tol: (optional) Tolerance

    Returns:
        x: An approximate solution to minimize ||Ax - b||_0.

    Raises:
        ValueError: An error occurred in evaluating the dimensionality of the input matrix A and vector b.

    """
    GAMMA_THR = 1e-8  # For numerical stability
    lambda_r1 = 1.0e-6  # regularization 1.0e-1
    lambda_x1 = 1.0e6  # source variance 1.0e6
    lambda_r2 = 1.0e-6  # regularization 1.0e-1
    lambda_x2 = 1.0e6  # source variance 1.0e6

    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    m, n = A.shape

    gamma = np.ones((m, 1))
    e_old = 1000. * np.ones((m, 1))
    ite = 0
    Gamma_e = []
    while ite < max_ite:
        Gamma_e = np.diag(gamma.T[0])
        C = Gamma_e + lambda_r1 * np.identity(m) + lambda_x1 * A.dot(A.T)
        g_e = np.linalg.solve(C, b)
        e = Gamma_e.dot(g_e)
        if np.linalg.norm(e - e_old) < tol:
            break
        e_old = e
        D = Gamma_e + lambda_r2 * np.identity(m) + lambda_x2 * A.dot(A.T)
        Ei = np.linalg.solve(D, Gamma_e)
        Sigma_e = Gamma_e - Gamma_e.dot(Ei)
        gamma = np.diag(e.dot(e.T)) + np.diag(Sigma_e) # extract diagonal factors
        gamma = np.array([np.maximum(gamma, GAMMA_THR)]).T

    E = lambda_r1 * np.identity(m) + Gamma_e + lambda_x1 * A.dot(A.T)
    Xi = np.linalg.solve(E, b)
    x = lambda_x1 * A.T.dot(Xi)

    return x


def sparse_bayesian_learning(A, b, max_ite=1000, tol=1.0e-8):
    """Derives approximate solution to minimize ||Ax - b||_0.

    Computes L0 residual minimization by sparse bayesian learning
    to derive an approximate solution for minimize ||Ax - b||_0.

    Args:
        A: A design matrix (numpy 2D array)
        b: A column vector as a numpy 2D array
        max_ite: (optional) Maximum number of iterations
        tol: (optional) Tolerance

    Returns:
        x: An approximate solution to minimize ||Ax - b||_0.

    Raises:
        ValueError: An error occurred in evaluating the dimensionality of the input matrix A and vector b.

    """
    GAMMA_THR = 1e-8 # For numerical stability
    lambda1 = 1.0 # Coefficient regularizer (default value is 1e-6)
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
    return x

def pos(A):
    """
    Returns positive elements of the numpy array.
    Turns negative elements into zeros.
    """
    return A * np.double(A > 0)


def neg(A):
    """
    Returns negative elements of the numpy array.
    Turns positive elements into zeros.
    """
    return A * np.double(A < 0)


def shrinkage(x, kappa):
    return pos(x - kappa) - pos(-x - kappa)


def rpca_svt(D, lambda_=None, max_ite=100000, tol=5.0e-4, tau=1.0e4, delta=0.9):
    """Computes Robust-PCA by SVT.
    Computes Robust-PCA matrix decomposition D = A + E by Singular Value Thresholding.
    ``A Singular Value Thresholding Algorithm for Matrix Completion,''
    J.-F. Cai, E. J. Candes, and Z. Shen, SIAM J. Optim., 20(4), 1956--1982 (2008)
    The decomposition is performed by solving the following problem:
        minimize_{A, E} ||A||_* + lambda_ * ||E||_1 s.t. D = A + E

    Args:
        D: Input matrix to be decomposed
        lambda_: A weighting parameter for controlling the low-rankness and sparsity
        max_ite: (optional) Maximum number of iterations
        tol: (optional) Tolerance
        tau: (optional)
        delta: (optional)

    Returns:
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
        lambda_ = 1.0 / np.sqrt(m)
    Y = np.zeros((m, n))  # Lagrange multiplier
    A = np.zeros((m, n))  # low-rank structure
    E = np.zeros((m, n))  # error

    ite = 0
    while ite < max_ite:
        ite = ite + 1
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        A = np.dot(U, np.dot(np.diag(pos(np.subtract(S, tau))), Vt))
        E = np.sign(Y) * pos(np.subtract(np.fabs(Y), lambda_ * tau))
        M = D - A - E
#        rankA = np.sum(np.diag((np.subtract(S, tau)))>0)  # for debugging
#        cardE = np.sum(np.fabs(E)>0)  # for debugging
        Y = Y + np.multiply(M, delta)
        if np.linalg.norm(D - A - E, ord='fro')/np.linalg.norm(D, 'fro') < tol or ite > max_ite:
            return A, E, ite
    warnings.warn("Exceeded the maximum number of iterations.")
    return A, E, ite


def rpca_inexact_alm(D, lambda_=None, max_ite=1000, tol=1.0e-6):
    """Computes Robust-PCA by inexact ALM.

    Computes Robust-PCA matrix decomposition D = A + E by the inexact augmented Lagrangian multiplier method.
    ``The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices,''
    Z. Lin, M. Chen, L. Wu, and Y. Ma (UIUC Technical Report UILU-ENG-09-2215, November 2009).
    The decomposition is performed by solving the following problem:
        minimize_{A, E} ||A||_* + lambda_ * ||E||_1 s.t. D = A + E

    Args:
        D: Input matrix to be decomposed
        lambda_: A weighting parameter for controlling the low-rankness and sparsity
        max_ite: (optional) Maximum number of iterations
        tol: (optional) Tolerance

    Returns:
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