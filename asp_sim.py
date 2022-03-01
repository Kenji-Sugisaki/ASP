import os
import socket
import openfermion
import numpy as np
import scipy as sp
import datetime
import time
import cirq
import itertools

from openfermion.config import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *
from openfermion.linalg import *
from cirq.ops import *
from cirq.circuits import *
from cirq import Simulator
from cirq.study import *
from numpy import dot, conjugate, zeros
from math import pi, sqrt, sin, ceil

"""
Quantum circuit simulation program for adiabatic state preparation
"""

# Simulation parameters
n_qubits          = 12
n_electrons       = 4
mapping_method    = "JWT"
s_squared_scaling = 0.5
trotter_term_ordering = "Magnitude"

# Simulation parameters
# Wave function configuration
integral_filename = "BeH2_R300_BS-UHF_STO-3G_LNOInt.out"
initial_config    = "BS"
num_bspair        = 1

# ASP conditions
# Weight strategy should be "Lin", "Sin", "Squ", "SinCub", or "Cub"
evolution_time       = 10
weight_strategy      = "Lin"
trotter_order        = 2
num_steps            = ceil(evolution_time*2.0)

# Add S^2 as a penalty term or not. Add 'Const' for constant penalty weight
use_s_squared        = False
if initial_config == "BS":
    use_s_squared = True
s2_strategy          = weight_strategy

#
host = socket.gethostname()
start_time = time.time()
current_datetime = datetime.datetime.now()
print(" Quantum simulation on {}".format(host)," starts at {}".format(current_datetime))

nmo = n_qubits // 2

additional_steps  = 10
preliminary_steps = 10

simulator = Simulator()
qubits = cirq.LineQubit.range(n_qubits+1)

strategies = ['Lin', 'Sin', 'Squ', 'SinCub', 'Cub', 'Const']
if weight_strategy not in strategies:
    raise ValueError(' Unidentified strategy for nonzero terms')


# Print computational conditions"
print("")
print(" +-------------------------------------------+")
print(" |        ADIABATIC STATE PREPARATION        |")
print(" |            OpenFermion & cirq             |")
print(" |                                           |")
print(" |   Coded by K. Sugisaki, Osaka City Univ   |")
print(" +-------------------------------------------+")
print("")
print(" <<< COMPUTATIONAL CONDITIONS >>>")
print("")
print(" == Molecules, Reference Wave Functions ==")
print(" Integral file         {}".format(integral_filename))
print(" Number of qubits      {}".format(n_qubits))
print(" Number of electrons   {}".format(n_electrons))
print(" Mapping method        {}".format(mapping_method))
print(" Initial configuration {}".format(initial_config))
if initial_config == "BS":
    print("  - Number of BS-pairs {}".format(num_bspair))
print("")
print(" == ASP Conditions ==")
print(" Evolution time        {}".format(evolution_time))
print(" Trotter order         {}".format(trotter_order))
print(" Trotter steps         {}".format(num_steps))
print(" Trotter term ordering {}".format(trotter_term_ordering))
print(" Weight strategy       {}".format(weight_strategy))
print(" S^2 penalty term      {}".format(use_s_squared))
if use_s_squared:
    print("  - S^2 scaling        {}".format(s_squared_scaling))
    print("  - S^2 strategy       {}".format(s2_strategy))
print(" =============================================")


#---------- FUNCTION GET_INTEGRALS ----------#
def get_integrals(filename, n_qubits):
    nlines = 0
    intdata = []
    for line in open(filename):
        items = line.split()
        intdata.append(items)
        nlines +=1

    nuclear_repulsion = 0
    oneint = zeros((n_qubits, n_qubits))
    twoint = zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for i in range(nlines):
        if len(intdata[i]) == 1:
            # Nuclear repulsion or frozen core energy
            nuclear_repulsion = float(intdata[i][0])
        elif len(intdata[i]) == 3:
            # One electron integrals
            iocc = int(intdata[i][0])
            avir = int(intdata[i][1])
            intvalue = float(intdata[i][2])
            oneint[iocc, avir] = intvalue
        elif len(intdata[i]) == 5:
            # Two electron integrals
            iocc = int(intdata[i][0])
            jocc = int(intdata[i][1])
            bvir = int(intdata[i][2])
            avir = int(intdata[i][3])
            intvalue = float(intdata[i][4])
            twoint[iocc, avir, jocc, bvir] = intvalue*2
    return nuclear_repulsion, oneint, twoint


#---------- FUNCTION GET_CLASSIFIED_HAMILTONIAN
def get_initial_hamiltonian(oneint, twoint, ini_occ, nqubits, nelec):
    #
    fock_ini = FermionOperator()
    corr_ini = FermionOperator()
    #
    for i in range(nqubits):
        for a in range(nqubits):
            if i == a and ini_occ[i] == 1:
                fock_ini += FermionOperator(((a, 1), (i, 0)), oneint[i, a])
            else:
                corr_ini += FermionOperator(((a, 1), (i, 0)), oneint[i, a])
    # Two electron terms:
    for i in range(nqubits):
        for j in range(nqubits):
            for a in range(nqubits):
                for b in range(nqubits):
                    if ini_occ[i] == 1 and ini_occ[j] == 1:
                        if (i == a and j == b) or (i == b and j == a):
                            fock_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                        0.5*twoint[i, a, j, b])
                        else:
                            corr_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                        0.5*twoint[i, a, j, b])
                    else:
                        corr_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                    0.5*twoint[i, a, j, b])

    return fock_ini, corr_ini


#---------- FUNCTION S_SQUARED_FERMION_DM ----------#
def s_squared_fermion_dm(n_qubits):
    # generate S^2 Fermionic operator in DM.
    """
    Notes:
    S(i,j)^2 = S_z(i)*S_z(j) + (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2
    """
    n_molorb = int(n_qubits / 2)
    s_squared_operator = FermionOperator()
     
    for iorb in range(n_molorb):
        ia = 2 * iorb
        ib  = 2 * iorb + 1
        for jorb in range(n_molorb):
            ja = 2 * jorb
            jb  = 2 * jorb + 1
            
            # S_z(i) * S_z(j) terms
            s_squared_operator +=  0.25 * FermionOperator(((ia, 1), (ia, 0), (ja, 1), (ja, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ia, 1), (ia, 0), (jb, 1), (jb, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ib, 1), (ib, 0), (ja, 1), (ja, 0)))
            s_squared_operator +=  0.25 * FermionOperator(((ib, 1), (ib, 0), (jb, 1), (jb, 0)))
            # (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2 terms
            s_squared_operator +=  0.50 * FermionOperator(((ia, 1), (ib, 0), (jb, 1), (ja, 0)))
            s_squared_operator +=  0.50 * FermionOperator(((ib, 1), (ia, 0), (ja, 1), (jb, 0)))    

    return s_squared_operator

#---------- IMPORTED FUNCTION: JW_NUMBER_INDICES ----------#
def jw_number_indices(n_electrons, n_qubits):
    """Return the indices for n_electrons in n_qubits under JW encoding

    Calculates the indices for all possible arrangements of n-electrons
        within n-qubit orbitals when a Jordan-Wigner encoding is used.
        Useful for restricting generic operators or vectors to a particular
        particle number space when desired

    Args:
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        indices(list): List of indices in a 2^n length array that indicate
            the indices of constant particle number within n_qubits
            in a Jordan-Wigner encoding.
    """
    occupations = itertools.combinations(range(n_qubits), n_electrons)
    indices = [sum([2 ** n for n in occupation])
               for occupation in occupations]
    return indices

#--------- IMPORTED FUNCTION: 
def jw_get_target_state_at_particle_number(sparse_operator, particle_number, ref_wf):
    """Compute ground energy and state at a specified particle number.

    Assumes the Jordan-Wigner transform. The input operator should be Hermitian
    and particle-number-conserving.

    Args:
        sparse_operator(sparse): A Jordan-Wigner encoded sparse matrix.
        particle_number(int): The particle number at which to compute the ground
            energy and states

    Returns:
        ground_energy(float): The lowest eigenvalue of sparse_operator within
            the eigenspace of the number operator corresponding to
            particle_number.
        ground_state(ndarray): The ground state at the particle number
    """
    num_states = 4

    n_qubits = int(np.log2(sparse_operator.shape[0]))

    # Get the operator restricted to the subspace of the desired particle number
    restricted_operator = jw_number_restrict_operator(sparse_operator,
                                                      particle_number,
                                                      n_qubits)

    # Compute eigenvalues and eigenvectors
    if restricted_operator.shape[0] - 1 <= 1:
        # Restricted operator too small for sparse eigensolver
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = sp.sparse.linalg.eigsh(restricted_operator,
                                                     k=num_states,
                                                     which='SA')
    # Expand the state
    target_so = 0
    target_st = 0
    for istate in range(num_states):
        curr_state = eigvecs[:, istate]
        expanded_state = zeros(2 ** n_qubits, dtype=complex)
        expanded_state[jw_number_indices(particle_number, n_qubits)] = curr_state
        overlap = dot(expanded_state, conjugate(ref_wf))
        sq_overlap = dot(overlap, conjugate(overlap)).real
        if sq_overlap > target_so:
            target_st = istate
            target_so = sq_overlap
    #
    target_ene = eigvals[target_st]
    target_state = eigvecs[:, target_st]
    target_wf = zeros(2 ** n_qubits, dtype=complex)
    target_wf[jw_number_indices(particle_number, n_qubits)] = target_state

    return target_ene, target_wf


#---------- FUNCTION TRANSFORM_QUBOP_TO_CIRQ ----------#
def transform_qubop_to_cirq(operator, theta):
    """
    Transform qubit operator to cirq circuit
    """
    len_op = len(operator)
    if len_op != 0:
        for iop in range(len_op):
            pauli_op = operator[iop]
            if pauli_op[1] == "X":
                yield H(qubits[pauli_op[0]])
            elif pauli_op[1] == "Y":
                yield rx(-pi/2).on(qubits[pauli_op[0]])
        #
        for iop in range(len_op-1):
            control_qubit = operator[iop][0]
            target_qubit = operator[iop + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])
        #
        yield rz(theta*2).on(qubits[operator[len_op-1][0]])
        #
        for iop in reversed(range(len_op-1)):
            control_qubit = operator[iop][0]
            target_qubit = operator[iop + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])

        for term in range(len_op):
            pauli_op = operator[term]
            if pauli_op[1] == "X":
                yield H(qubits[pauli_op[0]])
            elif pauli_op[1] == "Y":
                yield rx(pi/2).on(qubits[pauli_op[0]])


#---------- FUNCTION DISCARD_ZERO_IMAGINARY ----------#
def discard_zero_imaginary(qubit_operator):
    for key in qubit_operator.terms:
        qubit_operator.terms[key] = float(qubit_operator.terms[key].real)
    qubit_operator.compress()
    return qubit_operator


#---------- FUNCTION SUB_CIRCUITS ----------#
def sub_circuit(qubit_operator, trotter_order, trotter_term_ordering, time_for_single_trotter):
    qubit_operator = discard_zero_imaginary(qubit_operator)
    if trotter_term_ordering == "Magnitude":
        qubit_operator_sorted = sorted(list(qubit_operator.terms.items()),
                                       key=lambda x:abs(x[1]), reverse=True)
        num_qubit_terms = len(qubit_operator_sorted)
        if trotter_order == 2:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta)
            for iterm in reversed(range(num_qubit_terms)):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta)
        else:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta)
    else: 
        term_ordering = sorted(list(qubit_operator.terms.keys()))
        if trotter_order == 2:
            for op in term_ordering:
                theta = qubit_operator.terms[op] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta)
            for op2 in reversed(term_ordering):
                theta = qubit_operator.terms[op2] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op2, theta)
        else:
            for op in term_ordering:
                theta = qubit_operator.terms[op] * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta)
    yield X(qubits[-1])
    yield X(qubits[-1])


#---------- FUNCTION TRANSFORM_DM_AND_GSCM ----------#    
def transform_dm_and_gscm(nmo):
    """
    Transform direct mapping (DM) to generalized spin coordinate mapping (GSCM), and
        vice versa
                       DM      GSCM
     Doubly occupied   |11>    |01>
     Unoccupied        |00>    |00>
     spin-alpha        |10>    |10>
     spin-beta         |01>    |11>
    """
    for i in range(nmo):
        yield CNOT(qubits[2 * i + 1], qubits[2 * i])

#---------- FUNCTION CALC_SI_SQUARE_TERMS ----------#
def calc_si_square_terms(nmo, itime):
    """
    Calculate S(i)^2 terms. Eigenvalue of S(i)^2 is 3/4.
    ZPowGate gives exp(i*pi*theta) if the state is |1>. We want to obtain exp(-i*0.75*itime),
        and therefore theta = -0.75*itime/pi.
    """
    theta = -0.75 * itime / pi
    for i in range(nmo):
        yield ZPowGate(exponent=theta).on(qubits[2 * i])

#---------- FUNCTION N_SQUARE_TERMS ----------#
def calc_n_square_terms(nmo, itime):
    """
    Calculate N^2/4 terms for (i,j) pairs.
        instead of 0.25. 
    """
    theta = 0.25 * itime / pi
    for i in range(nmo):
        for j in range(nmo):
            if i != j:
                yield CZPowGate(exponent=theta).on(qubits[2 * i], qubits[2 * j])


#---------- FUNCTION GET_PIJ_CIRCUIT ----------#
def get_pij_circuit(imo, jmo, sim_time):
    theta = sim_time / pi
    i1 = 2 * imo
    i2 = 2 * imo + 1
    j1 = 2 * jmo
    j2 = 2 * jmo + 1
    #
    yield CCXPowGate(exponent=1.0).on(qubits[i1], qubits[j1], qubits[-1])
    yield CNOT(qubits[j2], qubits[i2])
    yield ZPowGate(exponent=-0.5*theta).on(qubits[-1])
    yield CCXPowGate(exponent=theta).on(qubits[-1], qubits[i2], qubits[j2])
    yield CNOT(qubits[j2], qubits[i2])
    yield CCXPowGate(exponent=1.0).on(qubits[i1], qubits[j1], qubits[-1])


#---------- FUNCTION LOOP_FOR_PIJ_TERMS ----------#
def loop_for_pij_terms(nmo, sim_time, trotter_order):
    if trotter_order == 2:
        for imo in range(nmo):
            for jmo in range(nmo):
                if imo != jmo:
                    yield get_pij_circuit(imo, jmo, sim_time*0.5)
        for imo in reversed(range(nmo)):
            for jmo in reversed(range(nmo)):
                if imo != jmo:
                    yield get_pij_circuit(imo, jmo, sim_time*0.5)
    else:
        for imo in range(nmo):
            for jmo in range(nmo):
                if imo != jmo:
                    yield get_pij_circuit(imo, jmo, sim_time)
# COPY END


#---------- FUNCTION GET_HARTREE_FOCK_CIRCUIT ----------#
def get_hartree_fock_circuit_jw(n_qubits, initial_occupation):
    for i_qubit in range(n_qubits):
        if initial_occupation[i_qubit] == 1:
            yield X(qubits[i_qubit])


def get_dummy(n_qubits):
    for i in range(n_qubits+1):
        yield X(qubits[i])
        yield X(qubits[i])



#---------- FUNCTION GET_HAMILTONIAN_WEIGHT ----------#
def get_hamiltonian_weight(i_step, num_steps, weight_strategy):
    curr_position = i_step / num_steps
    # Default is a linear function
    h_weight = curr_position
    #
    if weight_strategy == "Sin":
        h_weight = sin(pi * curr_position / 2.0)
    elif weight_strategy == "Squ":
        h_weight = 3 * curr_position**2 - 2 * curr_position**3
    elif weight_strategy == "SinCub":
        h_weight = (sin(pi * curr_position / 2.0))**3
    elif weight_strategy == "Cub":
        h_weight = 6 * curr_position**5 - 15 * curr_position**4 + 10 * curr_position**3
    elif weight_strategy == 'Const':
        h_weight = 1
    return h_weight


#---------- FUNCTION NORMALIZE_WAVE_FUNCTION ----------#
def normalize_wave_function(wave_function):
    wave_function_norm = dot(wave_function, conjugate(wave_function)).real
    wave_function = wave_function / sqrt(wave_function_norm)
    return wave_function


#---------- FUNCTION PRINT_WAVE_FUNCTION ----------#
def print_wave_function(wave_function):
    wfdim = len(wave_function)
    thresh = 0.0001
    for idet in range(wfdim):
        det = wave_function[idet]
        if dot(det, conjugate(det)) > thresh:
            print(det, '  {:13b}'.format(idet))



##########################################################################################
# set occupations
initial_occupation = [0.0]*n_qubits
if initial_config == "RHF":
    for i in range(n_electrons):
        initial_occupation[i] = 1
if initial_config == "BS":
    n_doc = n_electrons - (num_bspair * 2)
    if n_doc != 0:
        for iorb in range(n_doc):
            initial_occupation[iorb] = 1
    for bspair in range(num_bspair):
        initial_occupation[n_doc + 2*bspair] = 1
        initial_occupation[n_electrons + 1 + 2*bspair] = 1


s2_operator = s_squared_fermion_dm(n_qubits)
s2_operator_sparse = jordan_wigner_sparse(s2_operator, n_qubits = n_qubits+1)

# Hamiltonian term classification
nuclear_repulsion, oneint, twoint, = get_integrals(integral_filename, n_qubits)
fock_ops, corr_ops = get_initial_hamiltonian(oneint, twoint, initial_occupation,
                                             n_qubits, n_electrons)

# Define time-independent (initial) and time-dependent Hamiltonians
h_fock_jw = jordan_wigner(fock_ops)
h_corr_jw = jordan_wigner(corr_ops)
h_fock_sparse = jordan_wigner_sparse(fock_ops, n_qubits = n_qubits+1)
h_corr_sparse = jordan_wigner_sparse(corr_ops, n_qubits = n_qubits+1)

if use_s_squared:
    s_squared_hamiltonian = s_squared_scaling * s2_operator

full_hamiltonian_sparse = h_fock_sparse + h_corr_sparse

# Hartree-Fock energy
hf_pointer = 0
for i in range(n_qubits):
    if initial_occupation[i] == 1:
        hf_pointer += 2 ** (n_qubits - i)
hmat_dim = 2 ** (n_qubits+1)
hf_state = zeros(hmat_dim, dtype=np.complex64)
hf_state[hf_pointer] = 1.0+0.0j
hf_energy = expectation(full_hamiltonian_sparse, hf_state).real
print("\nE(HF) = {:.10f}".format(hf_energy.real)," Hartree")
# Full-CI energy
sparse_for_fci = full_hamiltonian_sparse + s2_operator_sparse
fci_energy, fci_state = jw_get_target_state_at_particle_number(sparse_for_fci, n_electrons, hf_state)
print("E(FCI_final) = {:.10f}".format(fci_energy.real)," Hartree")


#----------------------------------------------------------------------------------------------------#
# Generate quantum circuit for the time-independent Hamiltonian
time_for_single_trotter = evolution_time / num_steps


# Preliminary steps: Use time-independent Hamiltonian only
print("\n   Time  s(t)   s(S2)   E(ASP)/Hartree    <S^2>    E(Exact)/Hartree   |<Exact|Sim>|^2  |<FCI|Sim>|^2")
print("-----------------------------------------------------------------------------------------------------")

asp_wf_exact_curr = hf_state
asp_wf_sim_curr = hf_state

h_ins_jw = h_fock_jw
h_ins_sparse = h_fock_sparse

# Construct a quantum circuit for preliminary steps
h_ins_circuit = cirq.Circuit()
h_ins_circuit.append(sub_circuit(h_ins_jw, trotter_order, trotter_term_ordering, time_for_single_trotter))
h_ins_circuit.append(get_dummy(n_qubits))
for i_step in range(preliminary_steps):
    # Quantum circuit simulations of ASP time evoltution
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)
    # Exact ASP time evolution
    asp_wf_exact_next = sp.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter *
                                                       h_ins_sparse, asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)

    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp_sim = expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp_sim = expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact = expectation(h_ins_sparse, asp_wf_exact_next).real
    #
    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
    overlap_sim_fci = dot(fci_state, conjugate(asp_wf_sim_next))
    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
    print("  0.000  0.000  0.000    {:.10f}".format(e_asp_sim),"   {:.4f}".format(s2_asp_sim),\
          "  {:.10f}".format(e_asp_exact),"      {:.6f}".format(sq_overlap_sim_exact),\
          "       {:.6f}".format(sq_overlap_sim_fci))
    if i_step == 0:
        sq_overlap_sim_fci_ini = sq_overlap_sim_fci
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

# Time-dependent steps: Use both time-independent and time-dependent Hamiltonians
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
for i_step in range(num_steps):
    td_hamiltonian_weight = get_hamiltonian_weight(i_step+1, num_steps, weight_strategy)
    if use_s_squared:
        s2_hamiltonian_weight = get_hamiltonian_weight(i_step+1, num_steps, s2_strategy)
        

    # Calculate instantaneous eigenstate
    h_ins_jw = h_fock_jw + td_hamiltonian_weight * h_corr_jw
    h_ins_sparse = h_fock_sparse + td_hamiltonian_weight * h_corr_sparse
    if use_s_squared:
        h_ins_sparse += s2_hamiltonian_weight * s2_operator_sparse
    else:
        s2_hamiltonian_weight = 0
    # Quantum circuit simulations of ASP time evoltution
    h_ins_circuit = cirq.Circuit()
    h_ins_circuit.append(sub_circuit(h_ins_jw, trotter_order, trotter_term_ordering, time_for_single_trotter))
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)
    if use_s_squared:
        s2_prefactor = time_for_single_trotter * s2_hamiltonian_weight * s_squared_scaling
        s2_gscm_circuit = cirq.Circuit()
        s2_gscm_circuit.append(transform_dm_and_gscm(nmo))
        s2_gscm_circuit.append(calc_si_square_terms(nmo, s2_prefactor))
        s2_gscm_circuit.append(calc_n_square_terms(nmo, s2_prefactor))
        s2_gscm_circuit.append(loop_for_pij_terms(nmo, s2_prefactor, trotter_order))
        s2_gscm_circuit.append(transform_dm_and_gscm(nmo))
        #
        asp_wf_sim_int = asp_wf_sim_next
        asp_sim_result = simulator.simulate(s2_gscm_circuit, initial_state = asp_wf_sim_int)
        asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)
    # Exact ASP time evolution
    asp_wf_exact_next = sp.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse,
                                                          asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)
    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp = expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp = expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact = expectation(h_ins_sparse, asp_wf_exact_next).real
    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
    overlap_sim_fci = dot(asp_wf_sim_next, conjugate(fci_state))
    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
    #
    current_time = time_for_single_trotter * (i_step+1)
    print("  {:>.3f}".format(current_time)," {:.3f}".format(td_hamiltonian_weight),
          " {:.3f}".format(s2_hamiltonian_weight), "   {:.10f}".format(e_asp),
          "   {:.4f}".format(s2_asp),"  {:.10f}".format(e_asp_exact),\
          "      {:.6f}".format(sq_overlap_sim_exact),"       {:.6f}".format(sq_overlap_sim_fci))
    #
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

sq_overlap_sim_fci_fin = sq_overlap_sim_fci

if use_s_squared:
    s2_fin = 1.0
else:
    s2_fin = 0.0

# Additional steps to check convergence
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
for i_step in range(additional_steps):
    # Quantum circuit simulations of ASP time evoltutio
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)
    if use_s_squared:
        asp_wf_sim_int = asp_wf_sim_next
        asp_sim_result = simulator.simulate(s2_gscm_circuit, initial_state = asp_wf_sim_int)
        asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)
    # Exact ASP time evolution
    asp_wf_exact_next = sp.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse,
                                                          asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)
    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp_addi = expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp_addi = expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact_addi = expectation(h_ins_sparse, asp_wf_exact_next).real
    overlap_sim_exact_addi = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact_addi = dot(overlap_sim_exact_addi, conjugate(overlap_sim_exact_addi)).real
    overlap_sim_fci_addi = dot(fci_state, conjugate(asp_wf_sim_next))
    sq_overlap_sim_fci_addi = dot(overlap_sim_fci_addi, conjugate(overlap_sim_fci_addi)).real
    #
    print("   Addi  1.000    {:.3f}".format(s2_fin)," {:.10f}".format(e_asp_addi),"   {:.4f}".format(s2_asp_addi),\
          "  {:.10f}".format(e_asp_exact_addi),\
          "      {:.6f}".format(sq_overlap_sim_exact_addi),"       {:.6f}".format(sq_overlap_sim_fci_addi))
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

print("")
print(" SUMMARY OF THE QUANTUM CIRCUIT SIMULATION")
print("")
print("  E(ASP,Ini) = {:.10f}".format(hf_energy),"Hartree")
print("  E(ASP,Fin) = {:.10f}".format(e_asp),"Hartree")
print("  E(Full-CI) = {:.10f}".format(fci_energy),"Hartree")
print("")
print("  |<ASP,Ini|Full-CI>|^2 = {:.6f}".format(sq_overlap_sim_fci_ini))
print("  |<ASP,Fin|Full-CI>|^2 = {:.6f}".format(sq_overlap_sim_fci_fin))

elapsed_time = time.time() - start_time
print("\nNormal termination. Wall clock time is {}".format(elapsed_time) + "[sec]")

