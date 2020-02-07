# Transform AO integrals computed by GAMESS-US to MO integrals

import math
import numpy
import openfermion

from openfermion.ops import *
from openfermion.config import *
from openfermion.transforms import *
from openfermion.utils import *

# These values should be provided manually
gamess_logfile = "H2_R300_LMO_STO-3G_Int.out"
gamess_vecfile = "H2_R300_LMO.dat"

n_core  = 0
n_fzvir = 0

# nao, nelec, nucrep, aoint1e, aoint2e are obtained from GAMESS log file
integrals_1e = []
integrals_2e = []
int_thresh = 1.0e-8

print(' Open GAMESS log file {}'.format(gamess_logfile))

with open(gamess_logfile) as logfile:
    lines = logfile.readlines()
lines_strip = [line.strip() for line in lines]

# Read nao, nelec, nucrep
line_nao    = [line for line in lines_strip if 'NUMBER OF CARTESIAN' in line]
line_nelec  = [line for line in lines_strip if 'NUMBER OF ELECTRONS' in line]
line_nucrep = [line for line in lines_strip if 'THE NUCLEAR REPULSI' in line]

items = line_nao[0].split()
nao = int(items[7])
items = line_nelec[0].split()
nelec = int(items[4])
items = line_nucrep[0].split()
nucrep = float(items[5])
nmo = nao

print('   Number of atomic orbitals : {}'.format(nao))
print('   Number of electrons       : {}'.format(nelec))
print('   Nuclear repulsion energy  : {:.10f}'.format(nucrep))

# Read 1e Integral data
print('\n reading one-electron AO integrals...')
line_1eint_start = [i for i, line in enumerate(lines_strip) \
                    if 'BARE NUCLEUS HAMILTONIAN INTEGRALS (H=T+V)' in line]
line_1eint_end = [i for i, line in enumerate(lines_strip) \
                  if 'KINETIC ENERGY INTEGRALS' in line]
start_1eint = int(line_1eint_start[0])
end_1eint = int(line_1eint_end[0])

for rdint in range(start_1eint+2, end_1eint):
    integrals_1e.append(lines_strip[rdint].split())

# Read 2e Integral data
print(' reading two-electron AO integrals...')
line_2eint_start = [i for i, line in enumerate(lines_strip) \
                    if line.startswith('TWO ELECTRON INTEGRAL EVALUATION REQUIRES')]
line_2eint_end = [i for i, line in enumerate(lines_strip) \
                    if line.startswith('TOTAL NUMBER OF NONZERO TWO-ELECTRON INTEGRALS')]
start_2eint = int(line_2eint_start[0])
end_2eint = int(line_2eint_end[0])

for rdint in range(start_2eint+1, end_2eint):
    if not lines_strip[rdint].startswith('II,JST,KST,LST'):
        integrals_2e.append(lines_strip[rdint].split())


fzvir = nmo - n_fzvir


# Read MO vectors from modified GAMESS dat file
print('\n Open GAMESS vec file {}'.format(gamess_vecfile))
vec_nlines = 0
vectors = []
for line in open(gamess_vecfile, 'r'):
    vectors.append(line)
    vec_nlines += 1

line_for_onemo = -(-nao // 5)
if line_for_onemo * nmo != vec_nlines:
    raise ValueError('Vector dimension mismatch')

dim_movec = nmo * nao
mo_vectors = []
for imo in range(nmo):
    imo_vec = []
    for iline in range(line_for_onemo):
        current_line = imo * line_for_onemo + iline
        if iline != line_for_onemo - 1:
            read_iter = 5
        else:
            read_iter = nao % 5
            if read_iter == 0:
                read_iter = 5
#
        read_pointer = 5
        for iiter in range(read_iter):
            imo_vec.append(float(vectors[current_line][read_pointer:read_pointer+15]))
            read_pointer += 15
    mo_vectors.append(imo_vec)


### ONE-ELECTRON INTEGRALS TRANSFORMATION ###
print('\n AO-MO integral transformation started...')
aoint1e = numpy.zeros((nao, nao))

num_1eint_block = (nao // 5)
if nao % 5 != 0:
    num_1eint_block += 1

current_line = 2
for iblock in range(num_1eint_block):
    ao_ind = 5 * iblock
    len_ao_array = nao - ao_ind
    lentri = min(5, len_ao_array)

    for itri in range(lentri):
        for iiter in range(itri+1):
            aoint1e[ao_ind + itri, ao_ind + iiter] = float(integrals_1e[current_line][4+iiter])
        current_line += 1
    if lentri != len_ao_array:
        rest_ao_array = len_ao_array - lentri
        for irest in range(rest_ao_array):
            for iiter in range(5):
                aoint1e[ao_ind + irest + lentri, ao_ind + iiter] = float(integrals_1e[current_line][4+iiter])
            current_line += 1
        current_line += 3

for qao in range(nao):
    for pao in range(qao):
        aoint1e[pao, qao] = aoint1e[qao, pao]

moint1e = numpy.zeros((nmo, nmo))
int_am = numpy.zeros((nmo, nmo))

for pao in range(nao):
    for qao in range(nao):
        for jmo in range(nmo):
            int_am[pao, jmo] += aoint1e[pao, qao] * mo_vectors[jmo][qao]

for jmo in range(nmo):
    for pao in range(nao):
        for imo in range(nmo):
            moint1e[imo, jmo] += mo_vectors[imo][pao] * int_am[pao, jmo]

int_nlines = len(integrals_2e)
### TWO-ELECTRON INTEGRALS TRANSFORMATION ###
# Form aoint supermatrix
aoint2e = numpy.zeros((nao, nao, nao, nao))
for line in range(int_nlines):
    pao = int(integrals_2e[line][0]) -1
    qao = int(integrals_2e[line][1]) -1
    rao = int(integrals_2e[line][2]) -1
    sao = int(integrals_2e[line][3]) -1
    pqrs_ao = float(integrals_2e[line][5])
    aoint2e[pao, qao, rao, sao] = pqrs_ao
    aoint2e[qao, pao, rao, sao] = pqrs_ao
    aoint2e[pao, qao, sao, rao] = pqrs_ao
    aoint2e[qao, pao, sao, rao] = pqrs_ao
    aoint2e[rao, sao, pao, qao] = pqrs_ao
    aoint2e[sao, rao, pao, qao] = pqrs_ao
    aoint2e[rao, sao, qao, pao] = pqrs_ao
    aoint2e[sao, rao, qao, pao] = pqrs_ao
    if len(integrals_2e[line]) == 6:
        break
    pao = int(integrals_2e[line][6]) -1
    qao = int(integrals_2e[line][7]) -1
    rao = int(integrals_2e[line][8]) -1
    sao = int(integrals_2e[line][9]) -1
    pqrs_ao = float(integrals_2e[line][11])
    aoint2e[pao, qao, rao, sao] = pqrs_ao
    aoint2e[qao, pao, rao, sao] = pqrs_ao
    aoint2e[pao, qao, sao, rao] = pqrs_ao
    aoint2e[qao, pao, sao, rao] = pqrs_ao
    aoint2e[rao, sao, pao, qao] = pqrs_ao
    aoint2e[sao, rao, pao, qao] = pqrs_ao
    aoint2e[rao, sao, qao, pao] = pqrs_ao
    aoint2e[sao, rao, qao, pao] = pqrs_ao

# Two-electron integral transformation.
moint2e = numpy.zeros((nmo, nmo, nmo, nmo))

int_aaam = numpy.zeros((nao, nao, nao, nmo))
int_aamm = numpy.zeros((nao, nao, nmo, nmo))
int_ammm = numpy.zeros((nao, nmo, nmo, nmo))

# First transformation to generate (p,q,r,l)
for pao in range(nao):
    for qao in range(nao):
        for rao in range(nao):
            for sao in range(nao):
                for lmo in range(nmo):
                    int_aaam[pao, qao, rao, lmo] += aoint2e[pao, qao, rao, sao] * mo_vectors[lmo][sao]

# Second transformation to generate (p,q,k,l)
for pao in range(nao):
    for qao in range(nao):
        for lmo in range(nmo):
            for rao in range(nao):
                for kmo in range(nmo):
                    int_aamm[pao, qao, kmo, lmo] += int_aaam[pao, qao, rao, lmo] * mo_vectors[kmo][rao]

# Third transformation to generate (p,j,k,l)
for pao in range(nao):
    for kmo in range(nmo):
        for lmo in range(nmo):
            for qao in range(nao):
                for jmo in range(nmo):
                    int_ammm[pao, jmo, kmo, lmo] += mo_vectors[jmo][qao] * int_aamm[pao, qao, kmo, lmo]

# Fourth transformation to generate (i,j,k,l)
for jmo in range(nmo):
    for kmo in range(nmo):
        for lmo in range(nmo):
            for imo in range(nmo):
                for pao in range(nao):
                    moint2e[imo, jmo, kmo, lmo] += mo_vectors[imo][pao] * int_ammm[pao, jmo, kmo, lmo]


### GENERATE FERMIONOPERATOR FOR OPENFERMION ###
# Two-electron term

#Indices in moint2e is [imo^(1) jmo(1) kmo^(2) lmo(2)] where 1 and 2 specify the electron.
#Therefore, [i, i, j, j] is a Coulomb term and [i, j, j, i] is an Exchange.
#
#In OpenFermion, FermionOperator should be in the order [p^(1) q^(2) r(2) s(1)].
#Thus, the order [imo, kmo, lmo, jmo] should be used instead fo [imo, jmo, kmo, lmo].


# Frozen core approximation
fc_moint1e = numpy.zeros((nmo, nmo))
for imo in range(nmo):
    for jmo in range(nmo):
        oneint = 0
        for icore in range(n_core):
            oneint += 2.0 * moint2e[imo, jmo, icore, icore]
            oneint -= moint2e[imo, icore, icore, jmo]
        fc_moint1e[imo, jmo] = moint1e[imo, jmo] + oneint

if n_core == 0:
    print('\n MO integrals')
else:
    print('\n Frozen core MO integrals')
print()
fc_energy = nucrep
for icore in range(n_core):
    fc_energy += moint1e[icore, icore] + fc_moint1e[icore, icore]
print(fc_energy)
for imo in range(n_core, fzvir):
    for jmo in range(n_core, fzvir):
        if abs(fc_moint1e[imo,jmo]) > int_thresh :
            print((imo-n_core)*2,   (jmo-n_core)*2,   fc_moint1e[imo, jmo])
            print((imo-n_core)*2+1, (jmo-n_core)*2+1, fc_moint1e[imo, jmo])
for imo in range(n_core, fzvir):
    for jmo in range(n_core, fzvir):
        for kmo in range(n_core, fzvir):
            for lmo in range(n_core, fzvir):
                if abs(moint2e[imo, jmo, kmo, lmo]) > int_thresh:
                    print((imo-n_core)*2,   (kmo-n_core)*2,   (lmo-n_core)*2,   (jmo-n_core)*2,
                          0.5*moint2e[imo, jmo, kmo, lmo])
                    print((imo-n_core)*2+1, (kmo-n_core)*2,   (lmo-n_core)*2,   (jmo-n_core)*2+1,
                          0.5*moint2e[imo, jmo, kmo, lmo])
                    print((imo-n_core)*2,   (kmo-n_core)*2+1, (lmo-n_core)*2+1, (jmo-n_core)*2,
                          0.5*moint2e[imo, jmo, kmo, lmo])
                    print((imo-n_core)*2+1, (kmo-n_core)*2+1, (lmo-n_core)*2+1, (jmo-n_core)*2+1,
                          0.5*moint2e[imo, jmo, kmo, lmo])

