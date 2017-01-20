# Copyright (C) 2014-2015  BET Development Team

"""
This module contains functions choosing optimal QoIs to use in
the stochastic inverse problem.
"""

import numpy as np
from itertools import combinations
from bet.Comm import comm

def chooseOptQoIs(grad_tensor, qoiIndices=None, num_qois_return=None):
    """

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices)

    """
    (min_condnum_indices, optsingvals) = chooseOptQoIs_verbose( \
        grad_tensor, qoiIndices, num_qois_return)

    return min_condnum_indices

def chooseOptQoIs_verbose(grad_tensor, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None):
    """

    TODO:   This just cares about skewness, not sensitivity  (That is, we pass
            in normalized gradient vectors).  So we want to implement
            sensitivity analysis as well later.
            Check out 'magical min'.

            If a singular value is zero, we let the condition number be 
            1E9 at that point.  Possibly this should be a function of the
            dimension(?) so that we don't exclude a set simply because
            the vectors are linearly dependent at one point in :math:\Lambda,
            they could be much better in other regions.

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """
    num_xeval = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_qois_return is None:
        num_qois_return = Lambda_dim
    if num_optsets_return is None:
        num_optsets_return = 10

    # Find all posible combinations of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices),
                        num_qois_return)))
        print 'Possible sets of QoIs : ', qoi_combs.shape[0]
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and keep the set
    # that has the best skewness, i.e., smallest condition number
    condnum_indices_mat = np.zeros([num_optsets_return, num_qois_return + 1])
    condnum_indices_mat[:,0] = 1E11
    for qoi_set in range(len(qoi_combs)):
        singvals = np.linalg.svd(
            grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find the xeval that have atleast one zero sinular value
        indz = singvals[:,-1]==0
        indnz = singvals[:,-1]!=0

        current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1], \
                          axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

        if current_condnum < condnum_indices_mat[-1, 0]:
            condnum_indices_mat[-1, :] = np.append(np.array([current_condnum]),
                qoi_combs[qoi_set])
            condnum_indices_mat = condnum_indices_mat[condnum_indices_mat[:, 
                0].argsort()]
            #min_condnum_indices = np.append(np.array([current_condnum]),
            #    qoi_combs[qoi_set])
            optsingvals = singvals

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    min_condnum_indices = np.array(comm.gather(condnum_indices_mat, root=0))
    min_condnum_indices = min_condnum_indices.reshape(num_optsets_return * \
        comm.size, num_qois_return + 1)

    print min_condnum_indices

    # Find the minimum of the minimums
    if comm.rank == 0:
        min_condnum_indices = min_condnum_indices[min_condnum_indices[:, 
            0].argsort()]

    #print min_condnum_indices
    #print min_condnum_indices.shape

    min_condnum_indices = comm.bcast(min_condnum_indices, root=0)

    return (min_condnum_indices, optsingvals)

def find_bad_sets(grad_tensor, num_qois_return, qoiIndices=None):
    """

    TODO:   This just cares about skewness, not sensitivity  (That is, we pass
            in normalized gradient vectors).  So we want to implement
            sensitivity analysis as well later.
            Check out 'magical min'.

            If a singular value is zero, we let the condition number be 
            1E9 at that point.  Possibly this should be a function of the
            dimension(?) so that we don't exclude a set simply because
            the vectors are linearly dependent at one point in :math:\Lambda,
            they could be much better in other regions.

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """
    num_xeval = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_qois_return is None:
        num_qois_return = Lambda_dim

    # Find all n choose 2 pairs of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices), num_qois_return)))
        print 'Possible sets of QoIs : ', qoi_combs.shape
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and throw out one from each pair
    # that has global skewess>cond_tol
    cond_tol = 50
    bad_sets = np.zeros([1, num_qois_return])
    for qoi_set in range(len(qoi_combs)):
        singvals = np.linalg.svd(
            grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find the xeval that have atleast one zero sinular value
        indz = singvals[:,-1]==0
        indnz = singvals[:,-1]!=0

        # As it is with 1E9, if and singval is zero (for any xeval that is) we
        # throw out that pair  (unless we have BIG num_xeval)
        current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1], axis=0) +
            1E9 * np.sum(indz)) / singvals.shape[0]

        print bad_sets
        print bad_sets.shape
        if current_condnum > cond_tol:
            bad_sets = np.append(bad_sets, qoi_combs[qoi_set, :], axis=0)
            #optqoiIndices = qoi_combs[qoi_set]
            #optsingvals = singvals

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    bad_sets = np.array(comm.gather(bad_sets, root=0))

    print bad_sets
    print bad_sets.shape

    # Find the minimum of the minimums
    if comm.rank == 0:
        bad_sets = np.unique(bad_sets)
        #min_condnum = min_list[0]
        #optqoiIndices = min_list[1]

    #min_condnum = comm.bcast(min_condnum, root=0)
    bad_sets = comm.bcast(bad_sets, root=0)

    return bad_sets[1:]
