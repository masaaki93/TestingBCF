import numpy as np
import matplotlib.pyplot as plt
import input.expfit as expfit

from input.ESPRIT import ESPRIT
from scipy.integrate import quad, simpson
from scipy.optimize import minimize

"""
ℏ = 1

Step 1: Spectral density fitting
Fit the spectral density with the following ansatz
    Jmod(ω) = 4 Σ_{j=0}^{K_spec-1} [ 2 c[j].real Ω[j] γ[j] ω + c[j].imag (γ[j]^2 - Ω[j]^2 + ω^2) ω] / [(ω-Ω[j])^2 + γ[j]^2] / [(ω+Ω[j])^2 + γ[j]^2].
With this spectral density, the correlation function Lmod_ex(t) is given by
    Lmod_ex(t≧0) = Σ_{j=0}^{K_J-1} d_J[j] e^{- z_J[j] t} + Σ_{k=1}^{∞} b0_MT[k] e^{- γ0_MT[k] t}
where d_J, z_J, b0_MT, and γ0_MT can be determined from (arXiv_v1 Eq.(C2)).

Step 2: Matsubara contribution fitting
We approximate the Matsubara part of Lmod_ex(t≧0) as
    Lmod_ex(t≧0) = Σ_{j=0}^{K_J-1} d_J[j] e^{- z_J[j] t} + Σ_{j=0}^{K_MT-1} b_MT[j] e^{- γ_MT[j] t} + 2 η δ(t),
where
    b_MT ∈ R^{K_MT}
    γ_MT ∈ R_{>0}^{K_MT}
    η ∈ R_{>0}
The parameters b_MT, γ_MT, and η are determined to minimize a cost function (minimizing the deviation from the fluctuation-dissipation relation (MT_FD_fit) or that from ReL(t) (MT_ReL_fit)).
In this program, we chose the initial guess from the naive truncation
    b_MT = b0_MT
    γ_MT = γ0_MT
    η: Low-temperature correction, see [A.Ishizaki & Y.Tanimura, JPSJ 74, 3131 (2005).]

*1: LT = 0 or 1 => w or w/o low temperature correction.
*2: β = -1 (zero temperature) has not been implemented yet
"""

# ----------------------------------------
#   Step 1: Fitting the spectral density
# ----------------------------------------

def cμ_ESPRIT_fit(Lt_filename, Kj, data_ImLt = np.array([]), ImLt_filename = None, tol = 1e-6):

    if len(data_ImLt) == 0:
        if Lt_filename is not None:
            data_Lt = np.loadtxt(Lt_filename, dtype=complex)
            data_ImLt = np.column_stack((data_Lt[:, 0], data_Lt[:, 1].imag))
        elif ImLt_filename is not None:
            data_ImLt = np.loadtxt(ImLt_filename)

    # data
    t = data_ImLt[:,0].real
    t0 = t[0]
    dt = t[1]-t[0]
    ImLt = data_ImLt[:,1].real

    # optimization
    e, z, _ = ESPRIT(dt, ImLt, Kj, t[0])
    d = e * np.exp(z * t[0])

    # regularization
    d, z, _ = expfit.regularization(d, z, tol)

    if not np.all(z.real > 0):
        raise ValueError(f'Kj={Kj}: Lt.imag fitting is unstable (including a term exp(-zt) with z.real < 0)')

    c, μ = dz_to_cμ(d, z)
    # print(f'μ = {μ}')

    return c, μ

"""
Search the minimum Kj such that int_0^tf | Im[L(t)-L_{mod}(t)] | / int_0^tf | Im[L(t)] |  < relerr
"""
def cμ_ESPRIT_fit_relerr(Lt_filename, relerr = 1e-6, data_ImLt = np.array([]), ImLt_filename = None, tol = 1e-6):

    if len(data_ImLt) == 0:
        if Lt_filename is not None:
            data_Lt = np.loadtxt(Lt_filename, dtype=complex)
            data_ImLt = np.column_stack((data_Lt[:, 0], data_Lt[:, 1].imag))
        elif ImLt_filename is not None:
            data_ImLt = np.loadtxt(ImLt_filename)

    # data
    t = data_ImLt[:,0].real
    t0 = t[0]
    dt = t[1]-t[0]
    ImLt = data_ImLt[:,1].real

    Kj = 1; relerr1 = 1e+10
    while True:
        Kj += 1

        e, z, _ = ESPRIT(dt, ImLt, Kj, t[0])
        d = e * np.exp(z * t[0])

        relerr_Kj = simpson(np.abs(expfit.model_Lt(t, d, z) - ImLt), x = t) / np.max( np.abs(ImLt) ) / t[-1]
        # print(f'Kj = {Kj}, relerr = {relerr_Kj}')

        try:
            d, z, _ = expfit.regularization(d, z, tol)
        except ValueError:
            pass
        else:
            relerr_Kj_after = simpson(np.abs(expfit.model_Lt(t, d, z) - ImLt), x = t) / np.max( np.abs(ImLt) ) / t[-1]
            if np.all(z.real > 0) and relerr_Kj_after < relerr1:
                d1 = d; z1 = z; Kj1 = Kj; relerr1 = relerr_Kj_after

        if relerr_Kj < relerr:
            break

    c, μ = dz_to_cμ(d1, z1)
    # print(f'μ = {μ}')

    return Kj1, c, μ

# ----------------------------------------
#   Step 2: Cost function = deviation from the fluctuation dissipation theorem
#     cost(b_MT, γ_MT, η)
#     (opttype = integral) = ∫_{ω[0]}^{ω[-1]} dω [Jmod(ω) - tanh(βω/2) F[ReL](ω) ]^2
#     (opttype = points) = Σ_{i=0}^N-1 [Jmod(ω[i]) - tanh(βω[i]/2) F[ReL](ω[i]) ]^2
# which represents the deviation from the fluctuation-dissipantion relation.
# ----------------------------------------

def MT_FD_fit(c, μ, β, K_MT, ω, LT = 0, opttype = 'points'):

    d_J, z_J = cμ_to_aJzJ(c, μ, β)

    # Initial guess
    ps_init = naive_truncation_ps(c, μ, β, K_MT, LT)

    # optimization
    if opttype == 'integral':
        ps = minimize(cost_FD_integral, ps_init, args = (c, μ, β, K_MT, ω, LT), method = 'L-BFGS-B').x
    elif opttype == 'points':
        ps = minimize(cost_FD_points, ps_init, args = (c, μ, β, K_MT, ω, LT), method = 'L-BFGS-B', jac = True).x

    b_MT, γ_MT, η = from_ps(ps, K_MT, LT)

    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    print(f'cost_FD = {cost_FD_integral(to_ps(b_MT, γ_MT, η), c, μ, β, K_MT, ω, LT):.3e} @ K = {len(d)}')

    return d, z, η


def cost_FD_integral(ps, c, μ, β, K_MT, ω, LT):

    b_MT, γ_MT, η = from_ps(ps, K_MT, LT)

    d_J, z_J = cμ_to_aJzJ(c, μ, β)
    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    y = lambda x: (Jmod(x, c, μ) - np.tanh(β*x/2) * model_ReLω(x, d, z, η))**2

    return quad(y, ω[0], ω[-1])[0]

def cost_FD_points(ps, c, μ, β, K_MT, ω, LT):

    b_MT, γ_MT, η = from_ps(ps, K_MT, LT)

    d_J, z_J = cμ_to_aJzJ(c, μ, β)
    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    # return np.sum((Jmod(ω, c, μ) - np.tanh(β*ω/2) * model_ReLω(ω, d, z, η)) ** 2)

    common = - 4 * ((Jmod(ω, c, μ) - np.tanh(β*ω/2) * model_ReLω(ω, d, z, η)) * np.tanh(β*ω/2))[:,None]

    grad_b_MT = np.sum(common * γ_MT / (ω[:, None]**2 + γ_MT**2), axis = 0)
    grad_γ_MT = np.sum(common * b_MT * (ω[:, None]**2 - γ_MT**2) / (ω[:, None]**2 + γ_MT**2) ** 2, axis = 0)
    grad_η = np.sum(common, axis = 0)

    return np.sum((Jmod(ω, c, μ) - np.tanh(β*ω/2) * model_ReLω(ω, d, z, η)) ** 2), np.hstack([grad_b_MT, grad_γ_MT, LT * grad_η])


# ----------------------------------------
#   Step 2: Cost function = deviation from data Lt
# ----------------------------------------

def MT_ReLt_fit(Lt_filename, c, μ, β, K_MT, Lt_data = np.array([])):

    if len(Lt_data) == 0:
        Lt_data = np.loadtxt(Lt_filename, dtype=complex)

    # data
    t = np.real(Lt_data[:,0])
    Lt = Lt_data[:,1]

    d_J, z_J = cμ_to_aJzJ(c, μ, β)

    # Initial guess
    ps_init = naive_truncation_ps(c, μ, β, K_MT, 0)

    ps = minimize(cost_ReLt, ps_init, args = (c, μ, β, K_MT, t, Lt), method = 'L-BFGS-B', jac = True).x

    b_MT, γ_MT, η = from_ps(ps, K_MT, 0)

    # print(f'cost_Lt = {cost_ReLt(ps, c, μ, β, K_MT, t, Lt)}')
    # print()

    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    return d, z, η

def cost_ReLt(ps, c, μ, β, K_MT, t, Lt):

    b_MT, γ_MT, _ = from_ps(ps, K_MT, 0)

    d_J, z_J = cμ_to_aJzJ(c, μ, β)
    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    # return np.sum( (Lt.real - expfit.model_Lt(t, d, z).real) ** 2 )

    grad_b_MT = -2 * np.sum((Lt.real - expfit.model_Lt(t, d, z).real)[:, None] * np.exp(-γ_MT * t[:, None]), axis=0)
    grad_γ_MT = 2 * b_MT * np.sum((Lt.real - expfit.model_Lt(t, d, z).real)[:, None] * t[:, None] * np.exp(-γ_MT * t[:, None]), axis=0)

    return np.sum( (Lt.real - expfit.model_Lt(t, d, z).real) ** 2 ), np.hstack([grad_b_MT, grad_γ_MT, 0.])


# ----------------------------------------
#   Step 2: Pade decomposition (1 ≦ K_MT ≦ 5 for now)
# ----------------------------------------

def MT_Pade(c, μ, β, K_MT):

    d_J, z_J = cμ_to_aJzJ(c, μ, β)

    # [N-1/N] Pade coefficients (taken from [T. Ikeda & Y. Tanimura, JCTC 15, 2517 (2019).])
    if K_MT == 1:
        ν = (1/β) * np.array([7.745967])
        η = np.array([2.5])
    elif K_MT == 2:
        ν = (1/β) * np.array([6.305939, 19.499618])
        η = np.array([1.032824, 5.967176])
    elif K_MT == 3:
        ν = (1/β) * np.array([6.2832903, 12.9582867, 36.1192894])
        η = np.array([1.000227, 1.300914, 11.198859])
    elif K_MT == 4:
        ν = (1/β) * np.array([6.283185, 12.579950, 20.562598, 57.787940])
        η = np.array([1.000000, 1.015314, 1.905605, 18.079081])
    elif K_MT == 5:
        ν = (1/β) * np.array([6.283185, 12.566542, 19.004690, 29.579276, 84.536926])
        η = np.array([1.000000, 1.000262, 1.113033, 2.800147, 26.586558])
    else:
        raise ValueError(f'K_MT = {K_MT} has not been implemented yet')

    b_MT = 2*η/β * (1j * Jmod(1j*ν, c, μ)).real
    γ_MT = ν

    d = np.append(d_J, b_MT)
    z = np.append(z_J, γ_MT)

    return d, z, 0.



# ----------------------------------------
#   Model functions
# ----------------------------------------

def Jmod(ω, c, μ):

    γ = μ.real
    Ω = - μ.imag

    def Jmod_value(ω):
        return 4 * np.sum( (2 * c.real * Ω * γ * ω + c.imag * (γ**2 - Ω**2 + ω**2) * ω) / ((ω-Ω)**2 + γ**2) / ((ω+Ω)**2 + γ**2) )

    if isinstance(ω, (list, np.ndarray)):
        return np.vectorize(Jmod_value)(ω)
    else:
        return Jmod_value(ω)

def model_ImLt(t, c, μ):

    def ImLt_value(t):
        return - 2 * np.imag( np.sum(c * np.exp(- μ*t)) )

    if isinstance(t, (list, np.ndarray)):
        return np.vectorize(ImLt_value)(t)
    else:
        return ImLt_value(t)

"""
F[ReL](ω) = (F[L](ω) + F[L](-ω)) / 2
"""
def model_ReLω(ω, d, z, η):
    return (expfit.model_Lω(ω, d, z, η) + expfit.model_Lω(-ω, d, z, η)) / 2





# ----------------------------------------
#   Parameters
# ----------------------------------------

"""
Find d_J and z_J from c and μ
"""
def cμ_to_aJzJ(c, μ, β):

    d_J = np.array([])
    z_J = np.array([])

    for j in range(len(c)):
        if μ[j].imag == 0.:
            d_J = np.append(d_J, c[j]*(1/np.tanh(1j*β*μ[j]/2)-1) + np.conj(c[j]*(1/np.tanh(1j*β*μ[j]/2)+1)))
            z_J = np.append(z_J, μ[j])
        else:
            d_J = np.append(d_J, c[j]*(1/np.tanh(1j*β*μ[j]/2)-1))
            z_J = np.append(z_J, μ[j])
            d_J = np.append(d_J, np.conj(c[j]*(1/np.tanh(1j*β*μ[j]/2)+1)))
            z_J = np.append(z_J, np.conj(μ[j]))

    return d_J, z_J


"""
Find c and μ from d and z
"""
def dz_to_cμ(d, z):

    c = np.array([])
    μ = np.array([])

    i_taken = np.array([], dtype=int)
    for i in range(len(d)):
        if i in i_taken:
            continue

        if z[i].imag == 0:
            c = np.append(c,-1j*d[i].real/2)
            μ = np.append(μ,z[i])
            continue

        ibar = np.where(z == np.conjugate(z[i]))[0]
        if len(ibar) == 1:   # A pair is found
            i_taken = np.append(i_taken, ibar[0])
            c = np.append(c,-1j*d[i])
            μ = np.append(μ,z[i])
        elif len(ibar) == 0:  # No pair
            raise ValueError("z is not closed under complex conjugation.")
        elif len(ibar) > 1:
            raise ValueError("Error: duplication in z is found.")

    return c, μ

"""
From ps to b_MT, γ_MT, η.
"""
def from_ps(ps, K_MT, LT):

    b_MT = ps[:K_MT]
    γ_MT = np.abs(ps[K_MT:K_MT + K_MT])
    η = np.abs(ps[K_MT + K_MT]) * LT

    return b_MT, γ_MT, η

"""
From b_MT, γ_MT, η to ps.
If η < 0, we replace it by η = 0.
"""
def to_ps(b_MT, γ_MT, η):
    return np.hstack([b_MT, γ_MT, np.maximum(0.,η)])


"""
Compute ps associated with the naive truncation (b0_MT, γ0_MT, η0 from (arXiv_v1 Eq.(C2)))
"""
def naive_truncation_ps(c, μ, β, K_MT, LT):
    # set of Matsubara frequencies
    ν = 2 * np.pi * np.arange(1,K_MT+1) / β
    b0_MT = (2/β) * np.real( 1j * Jmod(1j*ν, c, μ) )
    γ0_MT = ν

    η0 = (2/β) * ( -4 * np.sum(c*(1 - 1j*β*μ/2/np.tanh(1j*β*μ/2)) / (2*(1j*μ)**2) ).imag - np.sum(1j * Jmod(1j*ν, c, μ) / ν).real)

    return to_ps(b0_MT, γ0_MT, η0*LT)





# ----------------------------------------
#   Plots
# ----------------------------------------

def plot_J_data(J_filename, c, μ, figure_name, ylog = False, save = True, show = False):

    data = np.loadtxt(J_filename)

    # data
    ω = data[:,0]
    J = data[:,1]

    plot_J(ω, J, c, μ, figure_name, ylog = ylog, save = save, show = show)

def plot_J(ω, J, c, μ, figure_name, ylog = False, save = True, show = False):

    Jmin = np.min(J)
    Jmax = np.max(J)
    ΔJ = (Jmax-Jmin) / 10

    plt.figure(figsize=(8, 6))
    plt.plot(ω,J,color='black',lw=3,label='Data')
    plt.plot(ω,Jmod(ω, c, μ),color='red',lw=1.5,label='Fit')
    if not ylog:
        plt.ylim(Jmin-ΔJ,Jmax+ΔJ)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.ylabel(r"$J(\omega)$", fontsize=15)
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(figure_name + '.png', format='png')
    if show:
        plt.show()
    plt.close()
















#
