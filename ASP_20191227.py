import os
import openfermion
import numpy
import scipy
import datetime
import time
import math
import itertools

from scipy.optimize import minimize
from numpy import dot, conjugate

from openfermion.config import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *

import cirq
import matplotlib.pyplot as plt

from cirq.ops import *
from cirq.circuits import *
from cirq.circuits import InsertStrategy
from cirq import Simulator
from cirq.study import *
from cirq.sim import *


from math import pi

"""
Quantum circuit simulation program for
Adiabatic state preparation

In this program I used the following notations:
Y_fci ... The Full-CI wave function of the final Hamiltonian
Y_ins ... The Full-CI wave function of the instantaneous Hamiltonian at time t
Y_asp  ... The wave function obtained from adiabatic state preparation

ASP aims to generate Y_asp having very large overlap with Y_fci
To check the adiabaticity, we can calculate overlap between Y_asp and Y_ins

The energy E(ASP) and E(Ins) during the adiabatic time evolutions are calculated
by using the instantaneous Hamiltonian. 

From literatures, it is desirable to make Hamiltonian time profiles so that
gradient dH(tau)/dtau = 0 for tau = 0 and 1. 
"""

# Simulation parameters (Fixed)
n_qubits          = 12
n_electrons       = 6
mapping_method    = "JWT"
s_squared_scaling = 0.5
trotter_term_ordering = "Magnitude"

# Simulation parameters (Should be varied)
# Wave function configuration
integral_filename = "N2_R300_STO-3G_FC4_UNOInt.out"
initial_config    = "RHF"
num_bspair        = 3

# ASP conditions
evolution_time    = 50.0
weight_strategy   = "Cubic"
trotter_order     = 2
num_steps         = math.ceil(evolution_time*2.0)
use_s_squared     = False


start_time = time.time()
current_datetime = datetime.datetime.now()
print(" Quantum circuit simulation starts on {}".format(current_datetime))

nmo = n_qubits // 2

additional_steps  = 10
preliminary_steps = 10
ethre = 1.0e-8


s2_strategy = weight_strategy

simulator = Simulator()
qubits = cirq.LineQubit.range(n_qubits+1)


# Print computational conditions"
print("\n +-------------------------------------------+")
print(" |        ADIABATIC STATE PREPARATION        |")
print(" |            OpenFermion & cirq             |")
print(" |                                           |")
print(" |   Coded by K. Sugisaki, Osaka City Univ   |")
print(" |      Version 1.2, December 27, 2019       |")
print(" +-------------------------------------------+")

print("\n <<< COMPUTATIONAL CONDITIONS >>>")
print("\n == Molecules, Reference Wave Functions ==")
print(" Integral file         {}".format(integral_filename))
print(" Number of qubits      {}".format(n_qubits))
print(" Number of electrons   {}".format(n_electrons))
print(" Mapping method        {}".format(mapping_method))
print(" Initial configuration {}".format(initial_config))
if initial_config == "BS":
    print("  - Number of BS-pairs {}".format(num_bspair))
print("\n == ASP Conditions ==")
print(" Evolution time        {}".format(evolution_time))
print(" Trotter order         {}".format(trotter_order))
print(" Trotter steps         {}".format(num_steps))
print(" Trotter term ordering {}".format(trotter_term_ordering))
print(" Weight strategy       {}".format(weight_strategy))
print(" S^2 penalty term      {}".format(use_s_squared))
if use_s_squared:
    print("  - S^2 scaling        {}".format(s_squared_scaling))
    print("  - S^2 strategy       {}".format(s2_strategy))
print("\n =============================================")


#---------- FUNCTION GET_INTEGRALS ----------#
def get_integrals(integral_filename, initial_occupation):
    """
    Obtain 1e and 2e integrals from file and transform them to QubitOperators
    """
    h1_fock = FermionOperator()
    h1_corr = FermionOperator()
    h2_fock = FermionOperator()
    h2_corr = FermionOperator()
    
    nlines = 0
    integral_data = []
    # Read integrals
    for line in open(integral_filename):
        items = line.split()
        integral_data.append(items)
        nlines += 1
    # Nuclear repulsion term
    h1_fock = FermionOperator((), float(integral_data[0][0]))
    # One and two-electron integrals
    one_electron_int = True
    for i in range(1, nlines):
        if one_electron_int:
            if len(integral_data[i]) == 5:
                one_electron_int = False
                cr1 = int(integral_data[i][0])
                cr2 = int(integral_data[i][1])
                an2 = int(integral_data[i][2])
                an1 = int(integral_data[i][3])
                int_value = float(integral_data[i][4])
                if initial_occupation[cr1] == 1 and initial_occupation[cr2] == 1:
                    if (cr1 == an1 and cr2 == an2) or (cr1 == an2) and (cr2 == an1):
                        h2_fock += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
                    else:
                        h2_corr += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
                else:
                    h2_corr += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
            else:
                cr1 = int(integral_data[i][0])
                an1 = int(integral_data[i][1])
                int_value = float(integral_data[i][2])
                if initial_occupation[cr1] == 1 and cr1 == an1:
                    h1_fock += FermionOperator(((cr1, 1), (an1, 0)), int_value)
                else:
                    h1_corr += FermionOperator(((cr1, 1), (an1, 0)), int_value)
        else:
            cr1 = int(integral_data[i][0])
            cr2 = int(integral_data[i][1])
            an2 = int(integral_data[i][2])
            an1 = int(integral_data[i][3])
            int_value = float(integral_data[i][4])
            if initial_occupation[cr1] == 1 and initial_occupation[cr2] == 1:
                if (cr1 == an1 and cr2 == an2) or (cr1 == an2 and cr2 == an1):
                    h2_fock += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
                else:
                    h2_corr += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
            else:
                h2_corr += FermionOperator(((cr1, 1), (cr2, 1), (an2, 0), (an1, 0)), int_value)
    return h1_fock, h1_corr, h2_fock, h2_corr

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


#---------- FUNCTION TRANSFORM_QUBOP_TO_CIRQ ----------#
def transform_qubop_to_cirq(operator, theta):
    """
    Transform qubit operator to cirq circuit
    """
    len_op = len(operator)
    theta = theta * 2

    if len_op == 0:
        pass
    else:
        for term in range(len_op):
            qub_operator = operator[term]
            if qub_operator[1] == "X":
                yield ([H(qubits[qub_operator[0]])])
            elif qub_operator[1] == "Y":
                yield (XPowGate(exponent=-0.5).on(qubits[qub_operator[0]]))

        for term in range(len_op-1):
            control_qubit = operator[term][0]
            target_qubit = operator[term + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])

        yield Rz(theta).on(qubits[operator[len_op-1][0]])

        for term in reversed(range(len_op-1)):
            control_qubit = operator[term][0]
            target_qubit = operator[term + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])

        for term in range(len_op):
            qub_operator = operator[term]
            if qub_operator[1] == "X":
                yield (H(qubits[qub_operator[0]]))
            elif qub_operator[1] == "Y":
                yield (XPowGate(exponent=0.5).on(qubits[qub_operator[0]]))

#---------- FUNCTION DISCARD_ZERO_IMAGINARY ----------#
def discard_zero_imaginary(qubit_operator):
    for key in qubit_operator.terms:
        qubit_operator.terms[key] = float(qubit_operator.terms[key].real)
    qubit_operator.compress()
    return qubit_operator


#---------- FUNCTION SUB_CIRCUITS ----------#
def sub_circuit(qubit_operator, trotter_order, trotter_term_ordering):
    qubit_operator = discard_zero_imaginary(qubit_operator)
    if trotter_term_ordering == "Magnitude":
        qubit_operator_sorted = sorted(list(qubit_operator.terms.items()),
                                       key=lambda x:abs(x[1]), reverse=True)
        num_qubit_terms = len(qubit_operator_sorted)
        if trotter_order == 2:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5
                yield transform_qubop_to_cirq(op, theta)
            for iterm in reversed(range(num_qubit_terms)):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5
                yield transform_qubop_to_cirq(op, theta)
        else:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1]
                yield transform_qubop_to_cirq(op, theta)
    else: 
        term_ordering = sorted(list(qubit_operator.terms.keys()))
        if trotter_order == 2:
            for op in term_ordering:
                theta = qubit_operator.terms[op] * 0.5
                yield transform_qubop_to_cirq(op, theta)
            for op2 in reversed(term_ordering):
                theta = qubit_operator.terms[op2] * 0.5
                yield transform_qubop_to_cirq(op2, theta)
        else:
            for op in term_ordering:
                theta = qubit_operator.terms[op]
                yield transform_qubop_to_cirq(op, theta)
    # Dummy
    yield X(qubits[-1])
    yield X(qubits[-1])


# COPIED FROM S^2 CALCULATOR
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
    theta = -0.75 * itime / math.pi
    for i in range(nmo):
        yield ZPowGate(exponent=theta).on(qubits[2 * i])

#---------- FUNCTION N_SQUARE_TERMS ----------#
def calc_n_square_terms(nmo, itime):
    """
    Calculate N^2/4 terms for (i,j) pairs.
        instead of 0.25. 
    """
    theta = 0.25 * itime / math.pi
    for i in range(nmo):
        for j in range(nmo):
            if i != j:
                yield CZPowGate(exponent=theta).on(qubits[2 * i], qubits[2 * j])


#---------- FUNCTION GET_PIJ_CIRCUIT ----------#
def get_pij_circuit(imo, jmo, sim_time):
    theta = sim_time / math.pi
    i1 = 2 * imo
    i2 = 2 * imo + 1
    j1 = 2 * jmo
    j2 = 2 * jmo + 1
    #
    yield CCXPowGate(exponent=1.0).on(qubits[i1], qubits[j1], qubits[-1])
    # Transform the basis from (00, 01, 10, 11) to (00, 11, 01, 10)
    yield CNOT(qubits[j2], qubits[i2])
#    yield CNOT(qubits[i2], qubits[j2])
    # Apply Pij operations
    yield ZPowGate(exponent=-0.5*theta).on(qubits[-1])
    yield CCXPowGate(exponent=theta).on(qubits[-1], qubits[i2], qubits[j2])
    # Back transfom the basis from (00, 11, 01, 10) to (00, 01, 10, 11)
#    yield CNOT(qubits[i2], qubits[j2])
    yield CNOT(qubits[j2], qubits[i2])
    #
    yield CCXPowGate(exponent=1.0).on(qubits[i1], qubits[j1], qubits[-1])


#---------- FUNCTION ROOP_FOR_PIJ_TERMS ----------#
def roop_for_pij_terms(nmo, sim_time, trotter_order):
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
    if weight_strategy == "Sinusoidal":
        h_weight = math.sin(math.pi * curr_position / 2.0)
    elif weight_strategy == "Square":
        h_weight = 3 * curr_position**2 - 2 * curr_position**3
    elif weight_strategy == "Sinusoidal-square":
        h_weight = (math.sin(math.pi * curr_position / 2.0))**2
    elif weight_strategy == "Sinusoidal-cubic":
        h_weight = (math.sin(math.pi * curr_position / 2.0))**3
    elif weight_strategy == "Cubic":
        h_weight = 6 * curr_position**5 - 15 * curr_position**4 + 10 * curr_position**3
    elif weight_strategy == "CubSin":
        h_weight = (6 * curr_position**5 - 15 * curr_position**4 + 10 * curr_position**3) * 0.5 +\
                   math.sin(math.pi * curr_position / 2.0) * 0.5
    return h_weight


#---------- FUNCTION NORMALIZE_WAVE_FUNCTION ----------#
def normalize_wave_function(wave_function):
    wave_function_norm = numpy.dot(wave_function, numpy.conjugate(wave_function)).real
    wave_function = wave_function / math.sqrt(wave_function_norm)
    return wave_function


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
        initial_occupation[2*bspair] = 1
        initial_occupation[n_electrons + 1 + 2*bspair] = 1



h1_fock, h1_corr, h2_fock, h2_corr = get_integrals(integral_filename, initial_occupation)
s2_operator = s_squared_fermion_dm(n_qubits)
s2_operator_sparse = jordan_wigner_sparse(s2_operator, n_qubits = n_qubits+1)

# Define time-independent (initial) and time-dependent Hamiltonians
time_independent_hamiltonian = h1_fock + h2_fock
time_dependent_hamiltonian = h1_corr + h2_corr

if use_s_squared:
    s_squared_hamiltonian = s_squared_scaling * s2_operator

full_hamiltonian = h1_fock + h2_fock + h1_corr + h2_corr
full_hamiltonian_sparse = jordan_wigner_sparse(full_hamiltonian, n_qubits = n_qubits+1)

# Hartree-Fock energy
hf_pointer = 0
for i in range(n_qubits):
    if initial_occupation[i] == 1:
        hf_pointer += 2 ** (n_qubits - i)
hmat_dim = 2 ** (n_qubits+1)
hf_state = numpy.zeros(hmat_dim, dtype=numpy.complex64)
hf_state[hf_pointer] = 1.0+0.0j
hf_energy = openfermion.expectation(full_hamiltonian_sparse, hf_state).real
print("\nE(HF) = {:.10f}".format(hf_energy.real)," Hartree")
# Full-CI energy
sparse_for_fci = full_hamiltonian_sparse + s2_operator_sparse
fci_energy, fci_state = jw_get_ground_state_at_particle_number(sparse_for_fci, n_electrons)
print("E(FCI_final) = {:.10f}".format(fci_energy.real)," Hartree")


#----------------------------------------------------------------------------------------------------#
# Generate quantum circuit for the time-independent Hamiltonian
time_for_single_trotter = evolution_time / num_steps


# Preliminary steps: Use time-independent Hamiltonian only
print("\n   Time  s(TD)   s(S2)   E(ASP)/Hartree    <S^2>    E(Exact)/Hartree   |<Exact|Sim>|^2  |<FCI|Sim>|^2")
print("-------------------------------------------------------------------------------------------------------")

asp_wf_exact_curr = hf_state
asp_wf_sim_curr = hf_state

h_ins = time_independent_hamiltonian
h_ins_jw = jordan_wigner(h_ins)
h_ins_sparse = jordan_wigner_sparse(h_ins, n_qubits = n_qubits+1)

# Construct a quantum circuit for preliminary steps
h_ins_circuit = cirq.Circuit()
h_ins_circuit.append(sub_circuit(h_ins_jw, trotter_order, trotter_term_ordering))
h_ins_circuit.append(get_dummy(n_qubits))
for i_step in range(preliminary_steps):
    # Quantum circuit simulations of ASP time evoltution
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state)
    # Exact ASP time evolution
    asp_wf_exact_next = scipy.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter *
                                                       h_ins_sparse, asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)

    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp_sim = openfermion.expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp_sim = openfermion.expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact = openfermion.expectation(h_ins_sparse, asp_wf_exact_next).real
    #
    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
    overlap_sim_fci = dot(fci_state, conjugate(asp_wf_sim_next))
    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
    print("  0.000  0.000   0.000  {:.10f}".format(e_asp_sim),"   {:.4f}".format(s2_asp_sim),\
          "  {:.10f}".format(e_asp_exact),"      {:.6f}".format(sq_overlap_sim_exact),\
          "       {:.6f}".format(sq_overlap_sim_fci))
    if i_step == 0:
        sq_overlap_sim_fci_ini = sq_overlap_sim_fci
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

# Time-dependent steps: Use both time-independent and time-dependent Hamiltonians
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
for i_step in range(num_steps):
    td_hamiltonian_weight = get_hamiltonian_weight(i_step+1, num_steps, weight_strategy)
    if use_s_squared:
        s2_hamiltonian_weight = get_hamiltonian_weight(i_step+1, num_steps, s2_strategy)
        

    # Calculate instantaneous eigenstate
    h_ins = time_independent_hamiltonian + td_hamiltonian_weight * time_dependent_hamiltonian
    h_ins_jw = jordan_wigner(normal_ordered(h_ins))
    if use_s_squared:
        h_ins += s2_hamiltonian_weight * s_squared_hamiltonian
    else:
        s2_hamiltonian_weight = 0
    h_ins_sparse = jordan_wigner_sparse(h_ins, n_qubits = n_qubits+1)
    # Quantum circuit simulations of ASP time evoltution
    h_ins_circuit = cirq.Circuit()
    h_ins_circuit.append(sub_circuit(h_ins_jw, trotter_order, trotter_term_ordering))
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state)
    if use_s_squared:
        s2_prefactor = time_for_single_trotter * s2_hamiltonian_weight * s_squared_scaling
        s2_gscm_circuit = cirq.Circuit()
        s2_gscm_circuit.append(transform_dm_and_gscm(nmo))
        s2_gscm_circuit.append(calc_si_square_terms(nmo, s2_prefactor))
        s2_gscm_circuit.append(calc_n_square_terms(nmo, s2_prefactor))
        s2_gscm_circuit.append(roop_for_pij_terms(nmo, s2_prefactor, trotter_order))
        s2_gscm_circuit.append(transform_dm_and_gscm(nmo))
        #
        asp_wf_sim_int = asp_wf_sim_next
        asp_sim_result = simulator.simulate(s2_gscm_circuit, initial_state = asp_wf_sim_int)
        asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state)
    # Exact ASP time evolution
    asp_wf_exact_next = scipy.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse,
                                                          asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)
    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp = openfermion.expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp = openfermion.expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact = openfermion.expectation(h_ins_sparse, asp_wf_exact_next).real
    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
    overlap_sim_fci = dot(fci_state, conjugate(asp_wf_sim_next))
    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
    #
    current_time = time_for_single_trotter * (i_step+1)
    print("  {:>.3f}".format(current_time)," {:.3f}".format(td_hamiltonian_weight),"  {:.3f}".format(s2_hamiltonian_weight),\
          " {:.10f}".format(e_asp),"   {:.4f}".format(s2_asp),"  {:.10f}".format(e_asp_exact),\
          "      {:.6f}".format(sq_overlap_sim_exact),"       {:.6f}".format(sq_overlap_sim_fci))
    #
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

sq_overlap_sim_fci_fin = sq_overlap_sim_fci

# Additional steps to check convergence
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
for i_step in range(additional_steps):
    # Quantum circuit simulations of ASP time evoltutio
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state)
    if use_s_squared:
        asp_wf_sim_int = asp_wf_sim_next
        asp_sim_result = simulator.simulate(s2_gscm_circuit, initial_state = asp_wf_sim_int)
        asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state)
    # Exact ASP time evolution
    asp_wf_exact_next = scipy.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse,
                                                          asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)
    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp_addi = openfermion.expectation(h_ins_sparse, asp_wf_sim_next).real
    s2_asp_addi = openfermion.expectation(s2_operator_sparse, asp_wf_sim_next).real
    e_asp_exact_addi = openfermion.expectation(h_ins_sparse, asp_wf_exact_next).real
    overlap_sim_exact_addi = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact_addi = dot(overlap_sim_exact_addi, conjugate(overlap_sim_exact_addi)).real
    overlap_sim_fci_addi = dot(fci_state, conjugate(asp_wf_sim_next))
    sq_overlap_sim_fci_addi = dot(overlap_sim_fci_addi, conjugate(overlap_sim_fci_addi)).real
    #
    print("    Addi  1.000   1.000  {:.10f}".format(e_asp_addi),"   {:.4f}".format(s2_asp_addi),\
          "  {:.10f}".format(e_asp_exact_addi),\
          "      {:.6f}".format(sq_overlap_sim_exact_addi),"       {:.6f}".format(sq_overlap_sim_fci_addi))
    asp_wf_exact_curr = asp_wf_exact_next
    asp_wf_sim_curr = asp_wf_sim_next

print("\n SUMMARY OF THE QUANTUM CIRCUIT SIMULATION")
print("\n  E(ASP,Ini) = {:.10f}".format(hf_energy),"Hartree")
print("  E(ASP,Fin) = {:.10f}".format(e_asp),"Hartree")
print("  E(Full-CI) = {:.10f}".format(fci_energy),"Hartree")
print("\n  |<ASP,Ini|Full-CI>|^2 = {:.6f}".format(sq_overlap_sim_fci_ini))
print("  |<ASP,Fin|Full-CI>|^2 = {:.6f}".format(sq_overlap_sim_fci_fin))



elapsed_time = time.time() - start_time
print("\nNormal termination. Wall clock time is {}".format(elapsed_time) + "[sec]")

