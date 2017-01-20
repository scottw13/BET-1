
import numpy as np
import matplotlib.pyplot as plt
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
from bet.sensitivity.gradients import *
import bet.util as util
import scipy.spatial as spatial


# Load data from files
lam_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
ref_lam = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
Q_ref = np.loadtxt("files/Q_ref.txt.gz") #reference QoI set
samples = np.loadtxt("files/samples.txt.gz") # uniform samples in parameter domain
data = np.loadtxt("files/data.txt.gz") # data from model

#samples = samples[:1000, :]
#data = data[:1000, :]

centers = samples[3,:]#[:10, :]
centers = util.fix_dimensions_vector_2darray(centers).transpose()
normalize = True


############################
#data = data[:, 0:2]

data = util.fix_dimensions_vector_2darray(util.clean_data(data))



Lambda_dim = samples.shape[1]
num_model_samples = samples.shape[0]
Data_dim = data.shape[1]

num_neighbors = 20
ep = 1.0
RBF = 'Gaussian'

# If centers is None we assume the user chose clusters of size
# Lambda_dim + 2
num_centers = 1#centers.shape[0]

rbf_tensor = np.zeros([num_centers, num_model_samples, Lambda_dim])
gradient_tensor = np.zeros([num_centers, Data_dim, Lambda_dim])
tree = spatial.KDTree(samples)

# For each centers, interpolate the data using the rbf chosen and
# then evaluate the partial derivative of that rbf at the desired point.
c = 0
# Find the k nearest neighbors and their distances to centers[c,:]
[r, nearest] = tree.query(centers[c, :], k=num_neighbors)
r = np.tile(r, (Lambda_dim, 1))

# Compute the linf distances to each of the nearest neighbors
diffVec = (centers[c, :] - samples[nearest, :]).transpose()

# Compute the l2 distances between pairs of nearest neighbors
distMat = spatial.distance_matrix(
    samples[nearest, :], samples[nearest, :])

# Solve for the rbf weights using interpolation conditions and
# evaluate the partial derivatives
rbf_mat_values = \
    np.linalg.solve(radial_basis_function(distMat, RBF),
    radial_basis_function_dxi(r, diffVec, RBF, ep) \
    .transpose()).transpose()

# Construct the finite difference matrices
rbf_tensor[c, nearest, :] = rbf_mat_values.transpose()

gradient_tensor = rbf_tensor.transpose(2, 0, 1).dot(data).transpose(1, 2, 0)

gradient_tensor[gradient_tensor < 1E-5] = 0

if normalize:
    # Compute the norm of each vector
    norm_gradient_tensor = np.linalg.norm(gradient_tensor, axis=2)

    # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
    norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0#sys.float_info[0]

    # Normalize each gradient vector
    gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
        (Lambda_dim, 1, 1)).transpose(1, 2, 0)

    gradient_tensor[gradient_tensor < 1E-5] = 0

