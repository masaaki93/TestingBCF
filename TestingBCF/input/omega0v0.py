import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
from input.exact import q2p2

"""
    Evalute ω0 and v0
"""

def ω0v0(Ω, C2, λ, β, η, J, ωmax):

    def compute_ω0(v0):
        λ_ = λ * v0**2
        ω0 = np.sqrt(λ_**2 + Ω**2) - λ_
        if ω0 == 0.:
            return Ω**2 / (2*λ_)
        else:
            return ω0

    def cost(v0):
        v0 = v0.squeeze()
        ω0 = compute_ω0(v0)
        M = 1/(ω0*v0**2)
        params = np.array([M, ω0, β])

        q2, _ = q2p2(params, tol = 1e-4, η = η, J = J, ωmax = ωmax)

        return (q2*v0**2 - C2)**2

    def ω0v0_method(method):
        v0 = np.sqrt(C2)   # initial guess
        result = minimize(cost, v0, method=method)
        v0 = np.abs(result.x.squeeze())
        return compute_ω0(v0), v0, np.sqrt(result.fun)/C2

    ω0, v0, rel_error = ω0v0_method('L-BFGS-B')
    if rel_error > 1e-2:    # we try Nelde-Mead if L-BFGS-B fails
        ω0, v0, rel_error = ω0v0_method('Nelder-Mead')
        if rel_error > 1e-2:
            raise ValueError(f'Optimization failed (take smaller tol): |v0^2 <q^2> -  C2|/C2 = {rel_error}')

    return ω0, v0, rel_error

"""
    Evalute Ω, C2, and p
"""

def system_transitions(λ, β, H_S, V_S, bin_size = 1e-12, pmax = 0.99):   # evaluate the Bohr frequencies Ω and C2(Ω) = tr_S[ (C_Ω + C_Ω^†)^2 ρ_{S,eq} ]

    H_S_eff = H_S - λ * V_S @ V_S

    if β > 0:
        ρ_S_eq = expm(- β * H_S_eff); ρ_S_eq /= np.trace(ρ_S_eq)
    elif β == -1: # zero temperature
        _, evec = eigh(H_S_eff)
        ρ_S_eq = np.outer(evec[:,0], np.conj(evec[:,0]))

    Ω_all, C2_all = Ω_C2(ρ_S_eq, H_S, V_S, bin_size)

    # taking into account up to pmax
    p_all = C2_all.copy(); p_all /= np.sum(p_all)
    idx = p_all.argsort()[::-1]

    Ω, C2, p = [], [], []
    for j in idx:
        Ω.append(Ω_all[j])
        C2.append(C2_all[j])
        p.append(p_all[j])
        if sum(p) > pmax:
            break
    Ω, C2, p = map(np.array, (Ω, C2, p))
    p /= np.sum(p)

    # print(f'length = {len(Ω)}')
    # print(f'Ω = np.array([{", ".join(str(val) for val in Ω)}])')
    # print(f'C2 = np.array([{", ".join(str(val) for val in C2)}])')
    # print(f'p = np.array([{", ".join(str(val) for val in p)}])')
    # print()

    return Ω, C2, p

def Ω_C2(ρ_S_eq, H_S, V_S, bin_size):  # Compute Bohr frequencies Ω and C2(Ω) = tr_S[ (C_Ω + C_Ω^†)^2 ρ_{S,eq} ] with jump operators C_Ω

    def dagger(A):  # A^†
        return A.conj().T

    def group_by_bin_span(data, bin_size):
        """
        Groups values so that all elements in a group are within `bin_size` of each other.
        Returns native Python number types and original indices.

        Parameters:
            data (array-like): Input values (list or array).
            bin_size (float): Maximum allowed span within a group.

        Returns:
            List of tuples: Each tuple is (group_values, group_indices),
            where group_values is a list of Python floats, and group_indices are ints.
        """
        data = np.array(data)
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]

        groups = []
        index_groups = []
        current_group = [sorted_data[0].item()]
        current_indices = [int(sorted_indices[0])]

        for val, idx in zip(sorted_data[1:], sorted_indices[1:]):
            test_group = current_group + [val.item()]
            if max(test_group) - min(test_group) <= bin_size:
                current_group.append(val.item())
                current_indices.append(int(idx))
            else:
                groups.append((current_group, current_indices))
                current_group = [val.item()]
                current_indices = [int(idx)]

        groups.append((current_group, current_indices))
        return groups

    # find the full set of Bohr frequencies
    eval, evec = eigh(H_S)

    Ω_matrix = eval[None, :] - eval[:, None]       # pairwise differences
    i, j = np.triu_indices(len(eval))               # upper triangle indices
    Ω_full = Ω_matrix[i, j]   # the full set of Bohr frequencies
    Ω_indices = np.column_stack((i, j))   # pairs of corresponding index

    idx = Ω_full.argsort()[::1]; Ω_full = Ω_full[idx]; Ω_indices = Ω_indices[idx,:] # ascending order

    # grouping with bin_size
    group = group_by_bin_span(Ω_full, bin_size); Number_of_group = len(group)

    Ω = np.array([np.min(g[0]) for g in group])

    C2 = np.zeros_like(Ω)
    C_Ω = np.zeros_like(H_S).astype(np.complex128)
    for i in range(Number_of_group):
        indices_i = group[i][1]
        C_Ω = 0.
        for k in range(len(indices_i)):
            left_index = Ω_indices[indices_i[k]][0]; right_index = Ω_indices[indices_i[k]][1]

            if left_index == right_index:
                C_Ω += 0.5 * (dagger(evec[:,left_index]) @ V_S @ evec[:,right_index]) * np.outer(evec[:,left_index], evec[:,right_index].conj())
                # print(f'dephasing contribution @ {left_index} = {evec[:,left_index].conj().T @ V @ evec[:,right_index]}')
            elif left_index < right_index:
                C_Ω += (dagger(evec[:,left_index]) @ V_S @ evec[:,right_index]) * np.outer(evec[:,left_index], evec[:,right_index].conj())
            else:
                raise ValueError('something is wrong in grouping')
        C2[i] = np.trace((C_Ω+dagger(C_Ω))@(C_Ω+dagger(C_Ω))@ρ_S_eq).real

    return Ω, C2
