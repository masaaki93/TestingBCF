import numpy as np
from scipy.integrate import simpson
from scipy.integrate import quad
import matplotlib.pyplot as plt


# q = (a + a^†)/√2, p = i(a^† - a)/√2 (dimensionless)


#
# <q^2> and <p^2>
#

"""
    Computing the expectation value of q^2 and p^2 for the total Gibbs state from hat{η}(z)

    *1 hat{η}(z) can be provided. If not, it's computed using 'η_integral'
    *2 The computations involve the infinite series, Σ_{n=1}^∞ a_n. Denoting S_N = Σ_{n=1}^N a_n, we truncate at n = N if
        |a_N / S_N| < tol

    Args:
        params: 3-vector representing parameters: params = [M, ω0, β]
        - optional -
        tol: positive real number (default: 1e-10)
        η: function representing hat{η}(z) with z ∈ R_{> 0} (default: None)
        J: function representing the spectral density. Necessary if η = None (default: None)
        ωmax: positive real number representing the cutoff frequency. Necessary if η = None (default: None)
"""
def q2p2(params, tol = 1e-10, η = None, J = None, ωmax = None):

    if η == None:
        if J == None or ωmax == None:
            raise ValueError('ERROR: J and ωmax must be given when η is not given')
        def η(z):
            return η_integral(z, J, ωmax)

    M = params[0]
    ω0 = params[1]
    β = params[2]

    # Nonzero temperature
    if β > 0:

        q2 = 1 / (β * ω0)
        p2 = 1 / (β * ω0)

        def q2p2_series(k):
            νk = 2 * np.pi * k / β
            ζk = νk * η(νk).real / M
            q2k = 2 / (β * ω0) * ω0 ** 2 / (ω0 ** 2 + νk ** 2 + ζk)
            p2k = 2 / (β * ω0) * (ω0 ** 2 + ζk) / (ω0 ** 2 + νk ** 2 + ζk)
            return q2k, p2k

        k = 0
        while True:
            k += 1

            q2k, p2k = q2p2_series(k)
            q2 += q2k
            p2 += p2k

            tol_q2 = np.abs( q2k / q2 )
            tol_p2 = np.abs( p2k / p2 )

            if tol_q2 < tol and tol_p2 < tol:
                break

    # Zero temperature
    elif β == -1:

        q2 = 0
        p2 = 0

        def ζ0(x):
            return x * η(ω0 * x).real / (M * ω0)

        yq2 = lambda x: 1 / (x ** 2 + 1 + ζ0(x)) / np.pi
        yp2 = lambda x: (1 + ζ0(x)) / (x ** 2 + 1 + ζ0(x)) / np.pi

        def q2p2_series(k):
            return quad(yq2, (k-1), k)[0], quad(yp2, (k-1), k)[0]

        k = 0
        while True:
            k += 1

            q2k, p2k = q2p2_series(k)
            q2 += q2k
            p2 += p2k

            tol_q2 = np.abs( q2k / q2 )
            tol_p2 = np.abs( p2k / p2 )

            if tol_q2 < tol and tol_p2 < tol:
                break

    return q2, p2

"""
    q2p2 for various tol
"""
def q2p2_tols(params, tols, title, η = None, J = None, ωmax = None):

    idx = tols.argsort()[::-1]; tols = tols[idx]

    if η == None:
        if J == None or ωmax == None:
            raise ValueError('ERROR: J and ωmax must be given when η is not given')
        def η(z):
            return η_integral(z, J, ωmax)

    M = params[0]
    ω0 = params[1]
    β = params[2]

    with open(title + '_q2p2.csv', "w") as f:
        f.write(f'tol  :   <q^2>_eq   :   <p^2>_eq   ' + '\n')

    # Nonzero temperature
    if β > 0:

        q2 = 1 / (β * ω0)
        p2 = 1 / (β * ω0)

        def q2p2_series(k):
            νk = 2 * np.pi * k / β
            ζk = νk * η(νk).real / M
            q2k = 2 / (β * ω0) * ω0 ** 2 / (ω0 ** 2 + νk ** 2 + ζk)
            p2k = 2 / (β * ω0) * (ω0 ** 2 + ζk) / (ω0 ** 2 + νk ** 2 + ζk)
            return q2k, p2k

        k = 0
        for i in range(len(tols)):
            tol = tols[i]
            while True:
                k += 1

                q2k, p2k = q2p2_series(k)
                q2 += q2k
                p2 += p2k

                tol_q2 = np.abs( q2k / q2 )
                tol_p2 = np.abs( p2k / p2 )

                if tol_q2 < tol and tol_p2 < tol:
                    break

            with open(title + '_q2p2.csv', "a") as f:
                f.write(f'{tol}     {q2}     {p2}' + '\n')
            print(f'tol = {tols[i]}')

    # Zero temperature
    elif β == -1:

        q2 = 0
        p2 = 0

        def ζ0(x):
            return x * η(ω0 * x).real / (M * ω0)

        yq2 = lambda x: 1 / (x ** 2 + 1 + ζ0(x)) / np.pi
        yp2 = lambda x: (1 + ζ0(x)) / (x ** 2 + 1 + ζ0(x)) / np.pi

        def q2p2_series(k):
            return quad(yq2, (k-1), k)[0], quad(yp2, (k-1), k)[0]

        k = 0
        for i in range(len(tols)):
            tol = tols[i]
            while True:
                k += 1

                q2k, p2k = q2p2_series(k)
                q2 += q2k
                p2 += p2k

                tol_q2 = np.abs( q2k / q2 )
                tol_p2 = np.abs( p2k / p2 )

                if tol_q2 < tol and tol_p2 < tol:
                    break

            with open(title + '_q2p2.csv', "a") as f:
                f.write(f'{tol}     {q2}     {p2}' + '\n')
            print(f'tol = {tols[i]}')

"""
    q2p2 from η(t)

    Args:
        params: 3-vector representing parameters: params = [M, ω0, β]
        t: numpy array (linspace) representing the time arguments
        ηt: numpy array representing the friction kernel η(t) (= Δ(t)) such that η = [η(t[0]), η(t[1]), η(t[2]), ...]
        - optional -
        tol: positive real number (default: 1e-10)
"""
def q2p2_ηt(params, t, ηt, tol = 1e-10):

    # computing the Laplace transform (z ∈ R>0):
    def η(z):
        return simpson( ηt * np.exp(- z * t), x = t)

    return q2p2(params, tol = tol, η = η, J = None, ωmax = None)





#
# Cqq(t) and Cpp(t)
#

"""
    Computing the Fourier transform of Cqq(t) and Cpp(t) from η

    Args:
        params: 3-vector representing parameters: params = [M, ω0, β]
        ω: numpy array representing a set of ω points (ω = 0, ±ωmax will be excluded)
        title: characters representing the filename (title_corr_fourier.csv and title_corr_fourier.png will be generated)
        - optional -
        η: function representing hat{η}(z) with z ∈ R_{> 0} (default: None)
        J: function representing the spectral density. Necessary if η = None (default: None)
        ωmax: positive real number representing the cutoff frequency. Necessary if η = None (default: None)
"""
def CqqCpp_η(params, ω, title, η = None, J = None, ωmax = None):

    ω0 = params[1]
    β = params[2]

    ω = ω[ω != 0.]
    if not ωmax == None:
        ω = ω[ω != ωmax]
        ω = ω[ω != -ωmax]

    corr_fourier = np.zeros((2,len(ω)), dtype=np.float64)        # F[Cqq], F[Cpp]

    if β > 0:
        corr_fourier[0,:] = 2 / (1. - np.exp(- β * ω)) * ImCqq_fourier(params, ω, η = η, J = J, ωmax = ωmax)
        corr_fourier[1,:] = (ω/ω0)**2 * corr_fourier[0,:]
    elif β == -1:
        corr_fourier[0,:] = np.where(
                                ω > 0,
                                2 * ImCqq_fourier(params, ω, η = η, J = J, ωmax = ωmax),
                                0.  # zero if ω < 0
                            )
        corr_fourier[1,:] = (ω/ω0)**2 * corr_fourier[0,:]

    save_corr_fourier(ω, corr_fourier, title)

"""
    Computing iF[ImCqq] from η

    Args:
        params: 3-vector representing parameters such that params = [M, ω0, β] (or 2-vector representing parameters such that params = [M, ω0])
        ω: numpy array (linspace) representing the frequency arguments
        - optional -
        η: function representing hat{η}(z) with z ∈ R_{> 0} (default: None)
        J: function representing the spectral density. Necessary if η = None (default: None)
        ωmax: positive real number representing the cutoff frequency. Necessary if η = None (default: None)
"""
def ImCqq_fourier(params, ω, η = None, J = None, ωmax = None):  # iF[ImCqq]

    M = params[0]
    ω0 = params[1]

    if η == None:
        if J == None or ωmax == None:
            raise ValueError('ERROR: J and ωmax must be given when η is not given')
        def η(z):
            return η_integral(z, J, ωmax)

    return (ω0 / (ω0**2 - ω**2 - 1j * (ω/M) * η(- 1j*ω))).imag

"""
    Computing the Fourier transform of Cqq(t) and Cpp(t) using the half-Fourier transform of G+(t)

    Args:
        params: 3-vector representing parameters: params = [M, ω0, β]
        t: numpy array (linspace) representing the time arguments
        Gp: numpy array representing G+(t) such that Gp = [G+(t[0]), G+(t[1]), G+(t[2]), ...]
        ω: numpy array representing a set of ω points (ω = 0 will be excluded)
        title: characters representing the filename (title_corr_fourier.csv and title_corr_fourier.png will be generated)
"""
def CqqCpp_Gp(params, t, Gp, ω, title):

    ω0 = params[1]
    β = params[2]

    ω = ω[ω != 0.]

    dt = t[1] - t[0]

    corr_fourier = np.zeros((2,len(ω)), dtype=np.float64)        # F[Cqq], F[Cpp]

    if β > 0:
        for i in range(len(ω)):
            corr_fourier[0,i] = 2 * ω0 / (1. - np.exp(- β * ω[i])) * simpson( Gp * np.sin(ω[i]*t), x = t)
        corr_fourier[1,:] = (ω/ω0)**2 * corr_fourier[0,:]
    elif β == -1:
        for i in range(len(ω)):
            if ω[i] > 0:
                corr_fourier[0,i] = 2 * ω0 * simpson( Gp * np.sin(ω[i]*t), x = t)
            else:
                corr_fourier[0,i] = 0.
        corr_fourier[1,:] = (ω/ω0)**2 * corr_fourier[0,:]

    save_corr_fourier(ω, corr_fourier, title)


def save_corr_fourier(ω, corr_fourier, title):

    f = open(title + '_corr_fourier.csv', mode="w")
    f.write('omega  :   F[Cqq](omega)   :   F[Cpp](omega)   ' + '\n')

    for j in range(len(ω)):
        f.write('{:12.3e}'.format(ω[j]))
        for k in range(np.shape(corr_fourier)[0]):
            f.write('{:20.8e}'.format(corr_fourier[k,j]))
        f.write('\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ω, corr_fourier[0,:], label=r'$\mathcal{F}[C_{qq}](\omega)$', color='red', lw='2')
    plt.plot(ω, corr_fourier[1,:], label=r'$\mathcal{F}[C_{pp}](\omega)$', color='blue', lw='2')
    # plt.plot(ω, (corr_fourier[0,:] + corr_fourier[0,:][::-1])/2, label=r'$\mathcal{F}[{\rm Re} C_{qq}](\omega)$', lw='2')
    # plt.plot(ω, (corr_fourier[0,:] - corr_fourier[0,:][::-1])/2, label=r'$i\mathcal{F}[{\rm Im} C_{qq}](\omega)$', lw='2')
    plt.xlabel(r'$\omega$', fontsize=15)
    plt.ylabel('Correlation function', fontsize=15)
    plt.ylim(-0.1, np.max(corr_fourier[1,:])+0.1)
    # plt.ylim(-0.1, np.max(corr_fourier)+0.1)
    # plt.xlim(-5, 15)
    # plt.ylim(-0.1, 17)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(title + '_corr_fourier.png', format='png')
    plt.close()











#
# G+(t)
#

"""
    Computing G+(t) by numerically integrating the differential equation with the 4th order Runge-Kutta method

    Args:
        params: 3-vector representing parameters such that params = [M, ω0, β] (or 2-vector representing parameters such that params = [M, ω0])
        t: numpy array (linspace) representing the time arguments
        ηt: numpy array representing the friction kernel η(t) (= Δ(t)) such that η = [η(t[0]), η(t[1]), η(t[2]), ...]
        title: characters representing the filename (title_η_Gp.csv and title_η_Gp.png will be generated)
"""
def compute_Gp(params, t, ηt, title):

    M = params[0]
    ω0 = params[1]

    dt = t[1] - t[0]
    N = len(t)

    def EOM_Gp(G, Gd, i):
        integral_term = 0
        if i > 1:
            integral_term = simpson( ηt[i:0:-1] * Gpd[:i], x = t[:i])
        return Gd, - (dt * ηt[0] * Gd + integral_term) / M - ω0 ** 2 * G

    def RK4(i): # t[i] → t[i+1]
        Gprk1, Gpdrk1 = EOM_Gp(Gp[i], Gpd[i], i)
        Gprk2, Gpdrk2 = EOM_Gp(Gp[i] + (dt/2)*Gprk1, Gpd[i] + (dt/2)*Gpdrk1, i)
        Gprk3, Gpdrk3 = EOM_Gp(Gp[i] + (dt/2)*Gprk2, Gpd[i] + (dt/2)*Gpdrk2, i)
        Gprk4, Gpdrk4 = EOM_Gp(Gp[i] + dt*Gprk3, Gpd[i] + dt*Gpdrk3, i)

        Gp[i+1] = Gp[i] + (Gprk1 + Gprk4 + 2*(Gprk2 + Gprk3))* dt/6
        Gpd[i+1] = Gpd[i] + (Gpdrk1 + Gpdrk4 + 2*(Gpdrk2 + Gpdrk3))* dt/6

    Gp = np.zeros(N)
    Gpd = np.zeros(N)
    Gpd[0] = 1  # Initial condition Gp(0) = 1
    # Gp[0] is already 0

    f = open(title + '_ηt_Gp.csv', mode="w")
    f.write('t  :   eta(t)   :   G+(t)   ' + '\n')
    f.write('{:8.4f}'.format(t[0]) + '{:15.3e}'.format(ηt[0]) + '{:15.3e}'.format(Gp[0]) + '\n')
    # Numerical solution using Runge-Kutta method
    for i in range(N-1):
        RK4(i)
        f.write('{:8.4f}'.format(t[i+1]) + '{:15.3e}'.format(ηt[i+1]) + '{:15.3e}'.format(Gp[i+1]) + '\n')
        print(f'Gp computation: {i} / {N-1}')
    print()
    print()

    plt.figure(figsize=(8, 6))
    plt.plot(t, ηt, label = r"$\eta(t)$", color = "blue", lw = 2)
    plt.plot(t, Gp, label = r"$G_+(t)$", color = "red", lw = 2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$t$', fontsize=15)
    # plt.show()
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(title + '_ηt_Gp.png', format='png')
    plt.close()

    return Gp

def Gp_slowest_fastest_decay_rate(params, c, μ):

    M = params[0]
    ω0 = params[1]

    a = 4 * (c*np.conj(μ)/np.abs(μ)**2/M).imag
    b = 4 * (c*np.conj(μ)**2/np.abs(μ)**2/M).imag

    # Step 1: Construct D(z)
    D = np.array([1.0])  # start with polynomial 1
    for m in μ:
        factor = [1.0, 2*m.real, abs(m)**2]  # z^2 + 2Re(m)z + |m|^2
        D = np.polymul(D, factor)

    # Step 2: Construct (z^2 + omega0^2)*D(z)
    P = np.polymul([1, 0, ω0**2], D)

    # Step 3: Add the sum terms
    for i in range(len(μ)):
        # denominator factor for term i
        Di = np.array([1.0])  # start with 1
        for j, m in enumerate(μ):
            if j != i:
                factor = [1.0, 2*m.real, abs(m)**2]
                Di = np.polymul(Di, factor)
        # term polynomial: (a_i z^2 + b_i z)*Di
        term = np.polymul([a[i], b[i], 0], Di)
        P = np.polyadd(P, term)

    # Step 4: Solve for roots
    roots = np.roots(P); roots = roots[roots < 0.]

    slowest_decay_rate = - np.max(roots.real)
    fastest_decay_rate = - np.min(roots.real)

    if slowest_decay_rate <= 0:
        raise ValueError(f'G+(t) has a negative decay rate = {slowest_decay_rate}')

    return slowest_decay_rate, fastest_decay_rate


#
# Other functions
#

"""
    Computing the Laplace transform of the friction kernel (hat{η}(z)) for the spectral density J(ω) with a finite support (J(ω≧0) = 0 (ω>ωmax)).
    We employ the integral formulas in (arXiv_v1 Eq.(11)) and the integration is performed using scipy.integrate.quad.

    *1 The allowed arguments are
        z ∈ R_{≠ 0} (necessary for 'q2p2') or
        z = i Im(z) (necessary for 'ImCqq_fourier').

    Args:
        z: complex number(s) (can be numpy array or list) representing the argument(s)
        J: function representing the spectral density
        ωmax: positive real number representing the cutoff frequency
"""
def η_integral(z, J, ωmax):

    def value(z, xmin = 1e-10):
        if z != 0. and z.imag == 0:  # z ∈ R_{≠ 0}
            y = lambda x: J(ωmax*x) / x / (x**2 + (z/ωmax)**2)
            return (2*z/(np.pi*ωmax**2)) * quad(y, xmin, 1.)[0]
        elif z.real == 0 and np.abs(z) > 0 and np.abs(z) < ωmax: # z = i Im(z) (0 < |z| < ωmax)
            y = lambda x: J(ωmax*x) / x / (x + (np.abs(z)/ωmax))
            return J(np.abs(z)) / np.abs(z) + (2*z/(np.pi*ωmax**2)) * quad(y, xmin, 1., weight='cauchy', wvar=np.abs(z)/ωmax)[0]
        elif z.real == 0 and np.abs(z) >= ωmax: # z = i Im(z) (|z| >= ωmax)
            y = lambda x: J(ωmax*x) / x / (x**2 + (z/ωmax)**2)
            return (2*z/(np.pi*ωmax**2)) * quad(y, xmin, 1.)[0]
        else:
            raise ValueError(f'Error: z = {z} has not yet been implemented in the function η_integral')

    if isinstance(z, (list, np.ndarray)):  # Check if z is a list or NumPy array
        return np.vectorize(value)(z)
    else:
        return value(z)


"""
    Laplace transform of the friction kernel (hat{η}(z)) for the GMT (analytic expression)
"""
def η_cμ(c, μ):

    def η(z):

        def value(z):
            return - 2j * np.sum(c/μ/(z+μ) - np.conj(c)/np.conj(μ)/(z+np.conj(μ)))

        if isinstance(z, (list, np.ndarray)):
            return np.vectorize(value)(z)
        else:
            return value(z)

    return η


"""
    Compute the Laplace transform from η(t)
"""
def η_ηt(t, ηt):

    # computing the Laplace transform (z ∈ R>0):
    def η(z):
        return simpson( ηt * np.exp(- z * t), x = t)

    return η
