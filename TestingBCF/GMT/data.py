import numpy as np
import matplotlib.pyplot as plt

# bath parameters; Im[L(t≧0)] = - 2 Σ_j Im[ c[j] exp(- μ[j] t)]
SD_name = 'GMT'

# Drude: J(ω) = ξ γ^2 ω / (γ^2 + ω^2) (λ = ξ γ / 2)
γ = 1; ξ = 1
c1 = np.array([1j*ξ*γ**2/4])
μ1 = np.array([γ])

# Brownian: J(ω) = ξ γ^2 ωb^2 ω / ((ω^2 - ωb^2)^2 + (γ ω)^2)
ωb = 1; γ = 5; ξ = 1
if ωb >= γ/2:
    ω0 = np.sqrt(ωb**2-(γ/2)**2)
    c2 = np.array([ξ*γ*ωb**2/(4*ω0)])
    μ2 = np.array([γ/2-1j*ω0])
elif ωb < γ/2:
    Γ = np.sqrt((γ/2)**2-ωb**2)
    c2 = np.array([1j*ξ*γ*ωb**2/(8*Γ), - 1j*ξ*γ*ωb**2/(8*Γ)])
    μ2 = np.array([γ/2-Γ, γ/2+Γ])

c = np.hstack([c1, c2])
μ = np.hstack([μ1, μ2])

# c = np.array([1+1j, 1])
# μ = np.array([1+2j, 2+1j])     # Re(μ) > 0

λ = 2 * np.sum(c/μ).imag
β = 1     # inverse temperature (must be finite)

Number_of_Matsubara_terms = 1000000

#
# get
#

def get_λ():
    return λ

def get_β():
    return β

def get_title():
    return f'_β{β}'

#
# spectral density / correlation function
#

def J_(ω):

    γ = μ.real; Ω = - μ.imag
    def value(ω):
        return 4 * np.sum( (2 * c.real * Ω * γ * ω + c.imag * (γ**2 - Ω**2 + ω**2) * ω) / ((ω-Ω)**2 + γ**2) / ((ω+Ω)**2 + γ**2) )

    if isinstance(ω, (list, np.ndarray)):
        return np.vectorize(value)(ω)
    else:
        return value(ω)

def J(ω):   # ω can be negative too
    return J_(ω)

def Lω_exact(ω):
    result = np.where(
        ω == 0,
        (2 / β) * np.sum(4 * np.imag(c * np.conj(μ) ** 2) / (μ.imag ** 2 + μ.real ** 2) ** 2),  # This handles the case when ω == 0
        2 * J(ω) / (1 - np.exp(- β * ω))  # Otherwise
    )
    return result

def Lt_exact(t):
    L = 0.
    for j in range(len(c)):
        L += c[j] * (1/np.tanh(1j*β*μ[j]/2) - 1) * np.exp(- μ[j] * t) + np.conj(c[j] * (1/np.tanh(1j*β*μ[j]/2) + 1) * np.exp(- μ[j] * t))

    for k in range(Number_of_Matsubara_terms):
        ν = 2 * k * np.pi / β
        L += (2 / β) * np.real( 1j * J_(1j * ν) ) * np.exp(- ν * t)
    return L

def ηt(t): # ηt: friction kernel in the time domain. Evaluated analytically
    ηt = 0.
    for j in range(len(c)):
        ηt += 4 * np.imag(c[j] / μ[j] * np.exp(- μ[j] * t))
    return ηt

def get_η(): # η: friction kernel in the Laplace domain

    print('Analytic hat{η}(z) is used')
    print()

    def η(z):

        def value(z):
            return - 2j * np.sum(c/μ/(z+μ) - np.conj(c)/np.conj(μ)/(z+np.conj(μ)))

        if isinstance(z, (list, np.ndarray)):
            return np.vectorize(value)(z)
        else:
            return value(z)

    return η

#
# plot
#

def output_Lt():

    filename = f'{SD_name}/{get_title()}/data_time'

    t = np.arange(0, 5e+1, 1e-1)
    Lt = Lt_exact(t)
    f = open(filename + '.csv', mode="w")
    for l in range(len(t)):
        f.write('{:12.5e}'.format(t[l]) + '{:50.15e}'.format(Lt[l]) + '\n')
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

    ω = np.arange(-10, 40, 5e-2)
    Lω = Lω_exact(ω)
    f = open(filename + '.csv', mode="w")
    for l in range(len(ω)):
        f.write('{:12.5e}'.format(ω[l]) + '{:25.15e}'.format(Lω[l]) + '\n')
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
