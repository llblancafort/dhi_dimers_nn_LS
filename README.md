# dhi_dimers_nn_LS
Data set files:
QBF.out, QBF_DHI_only.out, QBF_DHICA_only.out - Data set files containing the descriptor and output values for all molecules.
Python scripts:
DT_G_rel.py - training of the FP-DT model for G_rel endpoint.
RF_E_S1.py - training of the FP-RF model for E_exc1 endpoint.
NN_log_f_S1_norm.py - training of the  FP-DNN model for log(f) of S1 endpoint.
To train the model for other endpoints than given, change the name of the Endpoint variable to the header of the column of interest in the data file.

