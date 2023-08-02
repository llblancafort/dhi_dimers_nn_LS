# dhi_dimers_nn_LS
Data set files:
dhi_dimer_data.dat - 11 column file with the names of the compounds and the energy and osc. strength endpoints.
cyc.ox1.ox2.dat - file with a list of Ox1 and Ox2 cyclic dimers.
Python scripts:
cyc_arom.ox1.ox2.py - generates the AR descriptor value for the Ox1 and Ox2 cyclic dimers of cyc.ox1.ox2.dat (difficult cases).
NN_dhi_input_generator.py - generates the input files for the training with the different descriptor options.
NN_G_rel.py - training of the NNs for the G_rel endpoint.
NN_E_S1.py - training of the NNs for the E_S1 endpoint. Can be adapted to the remaining E_Sn and f_Sn endpoints.

WORKFLOW:
(1) run cyc_arom.ox1.ox2.py script to generate the AR descriptor for difficult cases.
(2) run NN_dhi_input_generator.py script to prepare the input files.
(3) run the NN training with the desired setup, eg python NN_G_rel.py QBF.out
