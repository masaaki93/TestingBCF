import numpy as np
from scipy.special import gamma as Γ
import scipy.special as sp
from mpmath import zeta as ζ
import matplotlib.pyplot as plt

# bath parameters; J(ω) = (π/2) α ωc^{1-s} ω^s e^{-ω/ωc} = πλ/(Γ(s)) (ω/ωc)^s e^{-ω/ωc}
SD_name = 'expcutoff'

s = 1   # Ohmicity
ωc = 10  # cutoff frequency
# λ = 0.1; α = 2*λ/(Γ(s)*ωc)   # reorganization energy
α = 1; λ = α*Γ(s)*ωc/2   # reorganization energy
β = 1    # inverse temperature  # beta = -1 for the zero temperature

# x = ω / ωc
xzero = 1e-12 # x = 0 is replaced by xzero

#
# get
#

def get_λ():
    return λ

def get_β():
    return β

def get_ωc():
    return ωc

def get_title():
    return f'_β{β}_s{s}_ωc{ωc}_α{α}'
    # return f'_β{β}_s{s}_ωc{ωc}_λ{λ}'

#
# spectral density / correlation function
#

def J_(ω):   # ω >= 0 (faster)
    return (np.pi/2) * α * ωc**(1-s) * ω**s * np.exp(-ω/ωc)

def J(ω):   # ω can be negative too
    ω = np.asarray(ω, float)
    out = np.sign(ω) * J_(np.abs(ω))
    return out.item() if np.isscalar(ω) else out

def Lω_exact(ω):
    # Use numpy's where to handle the array input
    if β == -1:
        return np.where(
            ω >= 0,
            2 * J(ω),  # This handles the case when ω >= 0
            0.  # Otherwise
        )
    elif β > 0:
        return np.where(
            ω == 0,
            Lω_exact_0(),   # This handles the case when ω == 0
            2 * J(ω) / (1 - np.exp(- β * ω))  # Otherwise
        )

def Lω_exact_0():  # Lω_exact(ω = 0) for β > 0
    if s < 1:
        return 2 * J(ωc*xzero) / (1 - np.exp(- β * ωc*xzero))
    elif s == 1:
        return np.pi * α / β
    elif s > 1:
        return 0.

def Lt_exact(t):

    if β == -1:
        return α * ωc**2 / 2 * Γ(s+1) / (1 + 1j*ωc*t)**(s+1)

    elif β > 0:
        def Lt_exact_value(t):
            z = (1 - 1j*ωc*t) / (β*ωc)
            # return α * ωc**2 / 2 / (β*ωc)**(s+1) * Γ(s+1) * (ζ(s+1,np.conjugate(z)) + ζ(s+1,z+1))
            return α * ωc**2 / 2 / (β*ωc)**(s+1) * Γ(s+1) * complex(ζ(s+1,np.conjugate(z)) + ζ(s+1,z+1))

        if isinstance(t, (list, np.ndarray)):  # Check if u is a list or NumPy array
            return np.vectorize(Lt_exact_value)(t)
        else:
            return Lt_exact_value(t)

def ηt(t): # ηt: friction kernel in the time domain. Evaluated analytically
    return np.where(t == 0.,
                        2*λ,
            α * ωc * Γ(s) * np.real((1-1j*ωc*t)**s) / (1+(ωc*t)**2)**s)

def get_η(): # η: friction kernel in the Laplace domain

    if float(s) == 1.:

        print('Analytic hat{η}(z) is used')
        print()

        # In the Ohmic case, the analytic expression is availables
        def η(z):
            def value(u):
                def _E_part(u):

                    def _E(u):
                        if np.imag(u) == 0 and np.real(u) < 0:
                            return - sp.expi(- u)
                        else:
                            return sp.exp1(u)

                    return - α * (_E(1j*u) * np.exp(1j*u) - _E(-1j*u) * np.exp(-1j*u)) / 2j

                if np.real(u) > 0:
                    return _E_part(u)
                elif u == 0.:
                    return α * np.pi / 2
                elif np.real(u) == 0.:
                    return _E_part(u) + α * np.pi / 2 * np.exp(-np.abs(u))

            u = z / ωc
            if isinstance(u, (list, np.ndarray)):  # Check if u is a list or NumPy array
                return np.vectorize(value)(u)
            else:
                return value(u)
        return η

    else:
        η = None

#
# plot
#

def output_Lt():

    filename = f'{SD_name}/{get_title()}/data_time'

    # y = ωc * t
    y_max = 200
    dy = 1e-1
    y = np.arange(0, y_max, dy)
    t = y/ωc
    Lt = Lt_exact(t)
    f = open(filename + '.csv', mode="w")
    for l in range(len(t)):
        f.write('{:8.3f}'.format(t[l]) + '{:30.5e}'.format(Lt[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(t,Lt.real,color='red',lw=3,label='Real')
    plt.plot(t,Lt.imag,color='blue',lw=3,label='Imag')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$L(t)$", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()


def output_Lω():

    filename = f'{SD_name}/{get_title()}/data_frequency'

    # x = ω / ωc
    x_min = -30
    x_max = +30
    dx = 1e-2
    x = np.arange(x_min, x_max, dx)
    ω = ωc*x
    Lω = Lω_exact(ω)
    f = open(filename + '.csv', mode="w")
    for l in range(len(ω)):
        f.write('{:8.3f}'.format(ω[l]) + '{:15.5e}'.format(Lω[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ω,Lω,color='red',lw=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.ylabel(r"$\mathcal{F}[L](\omega)$", fontsize=15)
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()
