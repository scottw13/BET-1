

import numpy as np
import matplotlib.pyplot as plt
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
import scipy.io as sio
import bet.Comm as comm

# Load data from files
lam_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
ref_lam = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
Q_ref = np.loadtxt("files/Q_ref.txt.gz") #reference QoI set
samples = np.loadtxt("files/samples.txt.gz") # uniform samples in parameter domain
data = np.loadtxt("files/data.txt.gz") # data from model


centers=samples[:100, :]
'''
G = grad.calculate_gradients_rbf(samples, data, centers=samples, num_neighbors=20)

if comm.rank==0:
    # save the results
    mdict = dict()
    mdict['G'] = G
    sio.savemat('G100centers.mat', mdict)
'''
#best_sets = cQoI.chooseOptQoIs_large(Gfull)

matfile = sio.loadmat('G100centers.mat')
G = matfile['G']

condnum_mat = cQoI.chooseOptQoIs(G, num_qois_return=2)
