import numpy as np
from scipy.linalg import svd, pinv

"""
The Estimation of signal parameters via rotational invariance techniques (ESPRIT) algorithm

Reference: https://doi.org/10.1016/j.laa.2012.10.036
Daniel Potts and Manfred Tasche,
Parameter estimation for nonincreasing exponentialsums by Prony-like methods,
Linear Algebra Appl. 439, 1024–1039 (2013).

Data:
    F ∈ C^2N: F[n] = f(t[n]) (n = 0, 1, ..., 2N-1).
    with t[n] = t0 + n Δt (n = 0, 1, ..., 2N-1).

Model function to fit the data
    fmod(t) = Σ_{m=0}^{M-1} c[m] exp(- η[m] (t - t0)).
where
    c ∈ C^M,
    η ∈ C^M.
We determine c and η so that
    F[n] = fmod(t[n])
         = Σ_{m=0}^{M-1} c[m] (Z[m]) ** n,
where Z ∈ C^M is defined by
    Z[m] = exp(- η[m] Δt),
with which we have
    η[m] = - ln(Z[m]) / Δt,
and
    fmod(t) = Σ_{m=0}^{M-1} c[m] (Z[m]) ** ((t-t0)/Δt).

M: # of exponential terms
L: L ≦ 2N

Args:
    Δt: positive number
    F: vector
    M: integer
    t0: real number (default: 0.)

Returns:
    c: complex vector representing coefficient
    η: complex vector representing the decay/oscillator rate
    fmod: function fmod(t)

"""
def ESPRIT(Δt, F, M, t0):

    F = np.array(F)

    if len(F) % 2 != 0:
        raise ValueError(f"Data length (= {len(F)}) needs to be even")

    N = int(len(F) / 2)
    L = N

    # Hankel matrix
    H = np.zeros((2*N-L, L+1), dtype=np.complex128)
    for l in range(2*N-L):
        H[l,:] = F[l:l+L+1]

    U, S, W = svd(H)
    W0 = W[:M,:L]
    W1 = W[:M,1:L+1]

    Z = np.linalg.eigvals(pinv(W0.T)@W1.T)
    η = - np.log(Z) / Δt

    # Vandermond matrix
    V = np.zeros((2*N, M), dtype=np.complex128)
    V[0,:] = np.ones(M)
    for n in range(1,2*N):
        V[n,:] = V[n-1,:] * Z

    c, residuals, rank, s = np.linalg.lstsq(V, F, rcond=None)

    def fmod(t):

        def fmod_value(t):
            return np.sum(c * (Z ** ((t-t0)/Δt)))

        if isinstance(t, (list, np.ndarray)):
            return np.vectorize(fmod_value)(t)
        else:
            return fmod_value(t)

    return c, η, fmod
