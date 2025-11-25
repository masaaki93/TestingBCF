import numpy as np
from scipy.special import beta as B
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# bath parameters; J(ω) = (π/2) α ωc^{1-s} ω^s [1 - (ω/ωc)^2]^m θ(ωc-ω)
SD_name = 'circcutoff'

s = 1   # Ohmicity
m = 0.5  # cutoff smoothness
ωc = 10  # cutoff frequency
α = 0.1 ; λ = α*ωc*B(s/2,m+1)/4   # reorganization energy
β = -1    # inverse temperature  # beta = -1 for the zero temperature

ωzero = 1e-10 # ω = 0 is replaced by ωzero

# ω-integral for Lt_exact and ηt
ωintegral = np.linspace(ωzero, ωc, 1000)

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
    return f'_β{β}_s{s}_m{m}_ωc{ωc}_α{α}'

#
# spectral density / correlation function
#

def J_(ω):   # ω >= 0 (faster)
    ω = np.asarray(ω, float)
    mask = (ω >= 0) & (ω <= ωc)

    out = np.zeros_like(ω)
    ωm = ω[mask]

    term = 1 - (ωm/ωc)**2
    out[mask] = (np.pi/2) * α * ωc**(1-s) * ωm**s * term**m

    return out.item() if np.isscalar(ω) else out

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
        return 2 * J(ωzero) / (1 - np.exp(- β * ωzero))
    elif s == 1:
        return np.pi * α / β
    elif s > 1:
        return 0.

def Lt_exact(t):
    t_max = np.max(t)

    def Lt_exact_value(t):
        print(f'{t:.3f} / {t_max}')

        # simpson
        if β > 0:
            return (1/np.pi) * simpson(J(ωintegral) * (np.cos(ωintegral*t) / np.tanh(β*ωintegral/2) - 1j * np.sin(ωintegral*t)), x=ωintegral)
        elif β == -1:
            return (1/np.pi) * simpson(J(ωintegral) * np.exp(-1j*ωintegral*t), x=ωintegral)

    if isinstance(t, (list, np.ndarray)):  # Check if u is a list or NumPy array
        return np.vectorize(Lt_exact_value)(t)
    else:
        return Lt_exact_value(t)

def ηt(t): # ηt: friction kernel in the time domain. Evaluated numerically

    def ηt_value(t):
        # simpson
        return (2/np.pi) * simpson(J(ωintegral) / ωintegral * np.cos(ωintegral*t), x=ωintegral)

    if isinstance(t, (list, np.ndarray)):  # Check if u is a list or NumPy array
        return np.vectorize(ηt_value)(t)
    else:
        return ηt_value(t)

def get_η(): # η: friction kernel in the Laplace domain
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
    x_min = -3
    x_max = +3
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
