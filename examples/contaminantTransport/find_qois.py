
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
import numpy as np

# Load data from files
lam_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
ref_lam = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
Q_ref = np.loadtxt("files/Q_ref.txt.gz") #reference QoI set
samples = np.loadtxt("files/samples.txt.gz") # uniform samples in parameter domain
dataf = np.loadtxt("files/data.txt.gz") # data from model

G = grad.calculate_gradients_rbf(samples, dataf, centers=samples[:100, :], num_neighbors=20)

best_sets = cQoI.chooseOptQoIs_large(G)
