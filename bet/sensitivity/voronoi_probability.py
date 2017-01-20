
import numpy as np
import matplotlib.pyplot as plt
import bet.sensitivity.gradients as grad
from mpl_toolkits.mplot3d import Axes3D
import pylab
from scipy.spatial import ConvexHull
import itertools
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import bet.postProcess.plotP as plotP

import scipy.spatial as spatial

from scipy.spatial import Voronoi, voronoi_plot_2d

# Method for finite voronoi diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

###############################################################################

# parameter domain
lam_domain= np.array([[0.0, 1.0],
                      [0.0, 1.0]])

Lambda_dim = 2
Data_dim = 4
num_samples = 5e3
num_samples_forbin = 100
size_scatter = 0
invert_with = 2

# Pick a random matrix Q
#np.random.seed(4)
Q = 2 * np.random.random([Data_dim, Lambda_dim]) - 1
Q1norm = np.linalg.norm(Q, ord=1, axis=1)
Q = Q / np.tile(Q1norm, (Lambda_dim, 1)).transpose()
#Q = np.array([[0.5, 0.5], [0.5, -0.5], [0, 1]])
#Q = np.array([[1, 0], [0, 1]])
Data_dim = Q.shape[0]

# Set bin_size (or radius) and choose Q_ref to invert to middle of Lambda
bin_size = 0.2
bin_radius = 0.1
Q_ref = Q.dot(0.5 * np.ones(Lambda_dim)).transpose()

# Samples and data
samples = np.random.random([num_samples, Lambda_dim])
data = Q.dot(samples.transpose()).transpose()

# Find the data that lie in the hypercube about Q_ref, and set probabilities
# on the corresponding voronoi cells assuming monte carlo assumtion
distances = np.linalg.norm(data[:num_samples_forbin, :] - Q_ref, ord=np.inf, axis=1)
d_distr_samples = data[:num_samples_forbin, :invert_with]
d_distr_prob = np.zeros(d_distr_samples.shape[0])
d_distr_prob[distances < bin_radius] = 1.0
d_distr_prob = d_distr_prob / np.sum(d_distr_prob)

# Restrict the data to just Lambda_dim QoIs
data = data[:, :invert_with]

# Plot the uniform probability on the new 'bin'
'''
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(d_distr_samples[d_distr_prob > 0, 0], d_distr_samples[d_distr_prob > 0, 1], color='r')
plt.show()
'''

# Plot Vornoi diagram of data and bin
vor = Voronoi(d_distr_samples)
regions, vertices = voronoi_finite_polygons_2d(vor)

for i in range(d_distr_samples.shape[0]):
    polygon = vertices[regions[i]]
    if d_distr_prob[i] > 0:
        plt.fill(*zip(*polygon), alpha=0.5)
    else:
        plt.fill(*zip(*polygon), alpha=0.2, color='k')

plt.scatter(d_distr_samples[:,0], d_distr_samples[:,1], s = size_scatter)
plt.xlim(vor.min_bound[0], vor.max_bound[0])
plt.ylim(vor.min_bound[1], vor.max_bound[1])
plt.show()


# Plot Lambda Voronoi diagram

d_distr_samples_nonzero = d_distr_samples[d_distr_prob > 0]
tree = spatial.KDTree(d_distr_samples)

vor = Voronoi(samples)
regions, vertices = voronoi_finite_polygons_2d(vor)

for i in range(data.shape[0]):
    [r, nearest] = tree.query(data[i], k=1)
    polygon = vertices[regions[i]]
    if d_distr_prob[nearest] > 0:
        plt.fill(*zip(*polygon), alpha=0.5)
    else:
        plt.fill(*zip(*polygon), alpha=0.2, color='k')


plt.scatter(samples[:,0], samples[:,1], s = size_scatter)
#plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
#plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

plt.xlim(vor.min_bound[0], vor.max_bound[0])
plt.ylim(vor.min_bound[1], vor.max_bound[1])


##########################################################
# Plot contours of actualy solution to inverse problem
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pylab import *

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

Lambda_min = 0.0
Lambda_max = 1.0

q_ref = Q_ref

delta = 0.025
x = np.arange(Lambda_min, Lambda_max, delta)
y = np.arange(Lambda_min, Lambda_max, delta)
X, Y = np.meshgrid(x, y)

Z1 = Q[0, 0] * X + Q[0, 1] * Y
Z2 = Q[1, 0] * X + Q[1, 1] * Y

levels1 = np.arange(Q_ref[0] - bin_radius, Q_ref[0] + bin_radius, 0.19999)
levels2 = np.arange(Q_ref[1] - bin_radius, Q_ref[1] + bin_radius, 0.19999)

cset1 = contourf(X, Y, Z1, levels1, colors='b', alpha=.2)
cset2 = contourf(X, Y, Z2, levels2, colors='r', alpha=.2)

if Q.shape[0] >= 3:
    Z3 = Q[2, 0] * X + Q[2, 1] * Y
    levels3 = np.arange(Q_ref[2] - bin_radius, Q_ref[2] + bin_radius, 0.19999)
    cset3 = contourf(X, Y, Z3, levels3, colors='g', alpha=.2)
if Q.shape[0] >= 4:
    Z4 = Q[3, 0] * X + Q[3, 1] * Y
    levels4 = np.arange(Q_ref[3] - bin_radius, Q_ref[3] + bin_radius, 0.19999)
    cset4 = contourf(X, Y, Z4, levels4, colors='y', alpha=.2)


plt.title('Title')
plt.show()



'''
# Find the simple function approximation
(d_distr_prob, d_distr_samples, d_Tree) =\
    simpleFunP.uniform_hyperrectangle_binsize(data=data, Q_ref=Q_ref,
    bin_size=bin_size, center_pts_per_edge = 10)

d_distr_samples = data


distr_dist = np.linalg.norm(d_distr_samples - Q_ref, ord=np.inf, axis=1)
d_distr_prob[distr_dist > bin_radius] = 0
'''
#d_distr_prob = np.ones(9)


'''
# Calculate probablities making the Monte Carlo assumption
(P,  lam_vol, io_ptr) = calculateP.prob(samples=samples, data=data,
    rho_D_M=d_distr_prob, d_distr_samples=d_distr_samples)

percentile = 1.0
# Sort samples by highest probability density and find how many samples lie in
# the support of the inverse solution.  With the Monte Carlo assumption, this
# also tells us the approximate volume of this support.
(num_samples, P_high, samples_high, lam_vol_high, data_high) =\
    postTools.sample_highest_prob(top_percentile=percentile, P_samples=P,
    samples=samples, lam_vol=lam_vol,data = data,sort=True)

# Print the number of samples that make up the highest percentile percent
# samples and ratio of the volume of the parameter domain they take up
if comm.rank == 0:
    print (num_samples, np.sum(lam_vol_high))


(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P,
    samples = samples, lam_domain = lam_domain, nbins = [50, 50])


# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain,
    filename = "nonlinearMap", plot_surface=False, interactive=True)

'''





'''
plt.scatter(samples[:, 0], samples[:, 1])
plt.scatter(samples_in[:, 0], samples_in[:, 1], color='r')
plt.show()
'''


'''
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(data_in[:, 0], data_in[:, 1], color='r')
plt.show()
'''






