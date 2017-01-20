import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab
import bet.util as util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
impoer bet

Lambda_min = 0
Lambda_max = 1


Q = np.array([[.5, .5], [.5, -.5], [0, 1]])
q_ref = np.array([.5, 0, .5])
q_ref = util.fix_dimensions_vector_2darray(q_ref).transpose()

samples = np.random.random([10000, 2])

plt.scatter(samples[:, 0], samples[:, 1])
plt.show()

############################################

data = Q.dot(samples.transpose()).transpose()
