import numpy as np
from scipy.linalg import svd, eig


"""
The adaptive Antoulas Anderson (AAA or triple-A) algorithm

Reference: https://epubs.siam.org/doi/10.1137/16M1106122
Y. Nakatsukasa, O. Sète, and L. N. Trefethen,
The AAA algorithm for rational approximation,
SIAM J. Sci. Comput. {\bf 40}, A1494–A1522 (2018)

Data:
Z ∈ C^N: Z[n] = z_n (n = 0, 1, ..., N-1),
F ∈ C^N: F[n] = f(z_n) (n = 0, 1, ..., N-1).

Model function to fit the data
    fmod(z;M,w,Z_smpl) = Σ_{m=0}^{M-1} [ w[m]*f(Z_smpl[m]) / (z - Z_smpl[m]) ] / Σ_{m=0}^{M-1} [ w[m] / (z - Z_smpl[m]) ],
where
    M: positive integer,
    w ∈ C^M (weight),
    Z_smpl ∈ C^M (⊂ Z) (sample points).
For M = 0, we define fmod(z;0,*,*) = 0 (this is necessary for the 1st step).
Objective is to minimize the error (introduced below) with the least possible M.

This is an iterative algorithm. Let
    w_previous ∈ C^k,
    Z_smpl_previous = [z_{i0}, z_{i1}, ..., z_{ik-1}] ∈ C^k (⊂ Z),
be w and Z_smpl at the end of the k-th step and
    Z_rest_previous = Z/Z_smpl_previous,
be the rest (the set of points that haven't been sampled by the end of the k-th step). In this case, we have
    idx_smple = [i0, i1, ..., ik-1],
at the beginning of the (k+1)-th step. We then determine z_{ik} using the infinity norm
    z_ik = arg min_{z ∈ Z_rest_previous} | f(z) - fmod(z;k,w_previous,Z_smpl_previous) |.
The new idx_smple is given by
    idx_smple = [i0, i1, ..., ik-1, ik],
from which we can determine the rest at the (k+1)-th step.
    idx_rest = [j0, j1, ..., jN-k-1].
The new weight w is then determined by
    w = arg min_{w ∈ C^{k+1}, |w| = 1} Σ_{l=0}^{N-k-1} | f(z_jl) - fmod(z_jl;k+1,w,Z[idx_smpl]) | * D(z_jl;k+1,w),
with D(z;k+1,w) being the denominator of fmod(z;k+1,w,Z[idx_smpl])
    D(z;k+1,w) = Σ_{m=0}^{k} [ w[m] / (z - Z[idx_smpl[m]]) ].
Owing to this normalization, the minimization problem can be solved exactly using the SVD.

Using the resulting w and idx_smpl, we compute the absolute and relative errors using the Frobenius norm
    err_abs = np.sqrt( Σ_{l=0}^{N-k-1} | f(z_jl) - fmod(z_jl;k+1,w,Z[idx_smpl]) |^2 )
    err_rel = np.sqrt( Σ_{l=0}^{N-k-1} | f(z_jl) - fmod(z_jl;k+1,w,Z[idx_smpl]) |^2 ) / np.linalg.norm(F)
               = err_abs / np.linalg.norm(F)

The iteration is terminated either when all the points are sampled or when the followings are achieved.
    err_abs < ε_abs ∧ err_rel < ε_rel

Args:
    Z: vector
    F: vector
    idx_smpl_default: vector such that Z[idx_smpl_initial] are included in the sample by default. If given, the iteration starts from it (default: [])
    ε_abs: positive number (default: np.inf)
    ε_rel: positive number (default: 1e-2)
    print_err: Whether print the errors at each step or not (default: False)

Returns:
    idx_smpl: vector representing the incides of the resulting sample points
    w: vector representing the final weight
    fmod: function fmod(z) = fmod(z;len(idx_smpl),w,Z[idx_smpl])

"""
def AAA(Z, F, idx_smpl_default = [], ε_abs = np.inf, ε_rel = 1e-2, print_err = True):

    Z = np.array(Z)
    F = np.array(F)

    N = len(Z)
    if N != len(F):
        raise ValueError(f"Dimension mistamch: dim(Z) = {N} vs dim(F) = {len(F)}")

    """
    Given idx_smpl (assuming len(idx_smpl) = k+1) and idx_rest, find the weight w ∈ C^{k+1} and Fmod_rest = fmod(Z[idx_rest];k+1,w,Z[idx_smpl])
    Args:
        idx_smpl: vector
        idx_rest: vector
    Returns:
        w: vector
        Fmod_rest: vector
    """
    def find_w_Fmod_rest(idx_smpl,idx_rest):

        # Sf = np.diag(F[idx_smpl])
        # SF = np.diag(F[idx_rest])
        # C = 1 / (Z[idx_rest,None] - Z[None,idx_smpl])   # Cauchy matrix
        # A = SF @ C - C @ Sf    # Loewner matrix

        F_smpl = F[idx_smpl]
        F_rest = F[idx_rest]

        Z_smpl = Z[idx_smpl]
        Z_rest = Z[idx_rest]

        C = 1 / (Z_rest[:, None] - Z_smpl[None, :])  # shape: (len(idx_rest), len(idx_smpl))

        # Multiply each row of C by F_rest and each column of C by F_smpl
        A = (F_rest[:, None] * C) - (C * F_smpl[None, :])

        # Find w by SVD of A
        U, S, Vt = svd(A, full_matrices=False)
        V = Vt.T
        w = V[:, -1]

        # Find Fmod_rest using
        numerator = C @ (w * F[idx_smpl])
        denominator = C @ w
        Fmod_rest = numerator / denominator

        return w, Fmod_rest

    if len(idx_smpl_default) == 0:
        # Initialize
        idx_smpl = np.array([], dtype=np.int32)
        idx_rest = np.setdiff1d(np.arange(N), idx_smpl)
        # Fmod_rest = np.zeros_like(F)
        Fmod_rest = np.mean(F) * np.ones_like(F)
    else:
        idx_smpl = idx_smpl_default
        idx_rest = np.setdiff1d(np.arange(N), idx_smpl)
        _, Fmod_rest = find_w_Fmod_rest(idx_smpl,idx_rest)

    # Main loop
    while True:
        # (k+1)-th step
        k = len(idx_smpl)

        # Find ik (special treatment for k = 1)
        if k == 1:
            # D = np.abs(F[idx_rest] - Fmod_rest)
            # D = np.abs(D - np.max(D)) - 2 * D[0]
            # argmin = np.argmin(np.abs(D))
            # ik = idx_rest[argmin]
            ik = idx_smpl[0] - 1
        else:
            argmax = np.argmax(np.abs(F[idx_rest] - Fmod_rest))
            ik = idx_rest[argmax]

        # Find ik
        # argmax = np.argmax(np.abs(F[idx_rest] - Fmod_rest))
        # ik = idx_rest[argmax]

        idx_smpl = np.append(idx_smpl, ik)
        idx_rest = np.setdiff1d(np.arange(N), idx_smpl)

        # Find w and Fmod_rest
        w, Fmod_rest = find_w_Fmod_rest(idx_smpl,idx_rest)

        # Error estimation
        err_abs = np.linalg.norm(F[idx_rest] - Fmod_rest, ord=np.inf)
        err_rel = err_abs / np.linalg.norm(F, ord=np.inf)

        if print_err:
            print(f'{k+1} (/ {N}) points are sampled: err_abs = {err_abs:.5e} and err_rel = {err_rel:.5e}')

        if err_abs < ε_abs and err_rel < ε_rel:
            break

        if k+1 == N:
            raise ValueError("Fitting failed: All the points are sampled")

    Z_smpl = Z[idx_smpl]
    F_smpl = F[idx_smpl]

    def fmod(z):

        def fmod_value(z): # f_mod(z) = f(z) (z ∈ Zmod), = f_mod(z) (else)
            if np.isin(z, Z_smpl):
                index = np.where(Z_smpl == z)[0]
                return F_smpl[index]
            else:
                return np.sum((w*F_smpl)/(z-Z_smpl)) / np.sum(w/(z-Z_smpl))

        if isinstance(z, (list, np.ndarray)):
            return np.vectorize(fmod_value)(z)
        else:
            return fmod_value(z)

    return idx_smpl, w, fmod, err_abs, err_rel


"""
Find the poles, residues, and zeros of
    f(z) = Σ_{m=0}^{M-1} [ w[m]*fz[m] / (z - z[m]) ] / Σ_{m=0}^{M-1} [ w[m] / (z - z[m])s ]

Args:
    z: vector
    w: vector
    fz: vector
    f: function

Returns:
    pol: Poles of f(z)
    res: Residues of f(z)
    zer: Zeros of f(z)

"""
def AAA_prz(z, w, fz, f):

    M = len(z)

    B = np.eye(M+1)
    B[0, 0] = 0
    E = np.block([
        [0,         w.reshape(1, -1)],
        [np.ones((M, 1)), np.diag(z)]
    ])

    # Poles: generalized eigenvalues of E,B
    pol = gen_eig(E, B)
    pol = pol[np.isfinite(pol)]

    # Residues
    Nθ = 4
    ε = 1e-5
    dz = ε * np.exp(2j * np.pi * np.arange(Nθ) / Nθ)
    PD = pol[:, None] + dz[None, :]
    Rvals = f(PD)  # shape (len(pol),Nθ)
    res = (Rvals @ dz[:, None]) / Nθ
    res = res.flatten()

    # Zeros
    E = np.block([
        [0,               (w * fz).reshape(1, -1)],
        [np.ones((M, 1)), np.diag(z)]
    ])
    zer = gen_eig(E, B)
    zer = zer[np.isfinite(zer)]

    return pol, res, zer

def gen_eig(A, B):
    # Solve generalized eigenvalue problem A v = λ B v
    # Using scipy.linalg.eig(A,B)
    vals, _ = eig(A, B)
    return vals
