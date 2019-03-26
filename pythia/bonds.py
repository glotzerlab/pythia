"""
This module computes relatively simple descriptors based on
nearest-neighbor bonds, with few additional transformations.
"""

import numpy as np
import freud

from .internal import cite


def _nlist_nn_helper(fbox, positions, neighbors, rmax_guess, exclude_ii=None):
    if isinstance(neighbors, int):
        nneigh = freud.locality.NearestNeighbors(rmax_guess, neighbors)
        nneigh.compute(fbox, positions, positions, exclude_ii)
        neighbors = nneigh.nlist

    return neighbors


@cite('freud2016')
def normalized_radial_distance(box, positions, neighbors, rmax_guess=2.):
    """Returns the ratio of the euclidean distance of each near-neighbor
    to that of the nearest neighbor for each particle.
    """
    fbox = freud.box.Box.from_box(box)

    neighbors = _nlist_nn_helper(fbox, positions, neighbors, rmax_guess, True)

    rijs = positions[neighbors.index_j] - positions[neighbors.index_i]
    fbox.wrap(rijs)

    rs = np.linalg.norm(rijs, axis=-1)
    reference_rs = rs[neighbors.segments]
    normalization = np.repeat(reference_rs, neighbors.neighbor_counts)
    rs /= normalization

    # skip the shortest bond since that gets normalized down to 1
    return rs.reshape((positions.shape[0], -1))[:, 1:]


def _get_neighborhood_distance_matrix(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise distances between `r_j - r_i` and `r_k - r_i`
    for all neighbors j and k of each particle i.
    """
    fbox = freud.box.Box.from_box(box)

    neighbors = _nlist_nn_helper(fbox, positions, neighbors, rmax_guess, True)

    neighbor_indices = neighbors.index_j.reshape((positions.shape[0], -1))

    # (Np, Nn, Nn, 3) distance matrix
    rijs = positions[neighbor_indices[:, :, np.newaxis]] - \
        positions[neighbor_indices[:, np.newaxis, :]]
    fbox.wrap(rijs.reshape((-1, 3)))

    # (Np, Nn, Nn) distance matrix
    rs = np.linalg.norm(rijs, axis=-1)
    # (0, 0) should be ri - ri == 0; (1, 0) should be the actual
    # nearest neighbor distance
    normalization = rs[:, 1, 0]
    rs /= normalization[:, np.newaxis, np.newaxis]
    return rs


@cite('freud2016')
def neighborhood_distance_singvals(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise distances filled with `|r_k - r_j|`
    for all neighbors j and k(==j) of each particle i. Returns the
    singular values of this matrix to fix permutation invariance.
    """
    rs = _get_neighborhood_distance_matrix(box, positions, neighbors, rmax_guess)
    svals = np.linalg.svd(rs, compute_uv=False)
    return svals


@cite('freud2016')
def neighborhood_range_distance_singvals(box, positions, neigh_min, neigh_max, rmax_guess=2.):
    """Construct a matrix of pairwise distances filled with `|r_k - r_j|`
    for all neighbors j and k(==j) of each particle i, for a range of
    neighborhood sizes from neigh_min to neigh_max
    (inclusive). Returns the singular values of this matrix to fix
    permutation invariance.
    """
    result = []

    for neighbors in range(neigh_min, neigh_max + 1):
        result.append(neighborhood_distance_singvals(box, positions, neighbors, rmax_guess))

    return np.hstack(result)


@cite('freud2016')
def neighborhood_distance_sorted(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise distances filled with `|r_k - r_j|`
    for all neighbors j and k(==j) of each particle i. Returns the
    sorted contents of this matrix to fix permutation invariance.
    """
    rs = _get_neighborhood_distance_matrix(box, positions, neighbors, rmax_guess)
    rs = rs.reshape((rs.shape[0], -1))
    np.sort(rs)
    return rs


def _get_neighborhood_angle_matrix(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise angles between `r_j - r_i` and `r_k - r_i`
    for all neighbors j and k of each particle i.
    """
    fbox = freud.box.Box.from_box(box)

    neighbors = _nlist_nn_helper(fbox, positions, neighbors, rmax_guess)

    neighbor_indices = neighbors.index_j.reshape((positions.shape[0], -1))

    # (Np, Nn, 3) distance matrix
    rijs = positions[neighbor_indices] - positions[:, np.newaxis, :]
    fbox.wrap(rijs.reshape((-1, 3)))

    # (Np, Nn) distances
    rs = np.linalg.norm(rijs, axis=-1)
    rijs /= rs[:, :, np.newaxis]

    # (Np, Nn, Nn) dot products of distances
    dots = np.sum(rijs[:, :, np.newaxis]*rijs[:, np.newaxis, :], axis=-1)
    thetas = np.arccos(dots)
    thetas[np.isnan(thetas)] = 0
    return thetas


@cite('freud2016')
def neighborhood_angle_singvals(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise angles between `(rk - ri)` and `(rj -
    ri)` for all neighbors j and k(==j) of each particle i, for a
    particular number of neighbors. Returns the singular values of
    this matrix to fix permutation invariance.
    """
    thetas = _get_neighborhood_angle_matrix(box, positions, neighbors, rmax_guess=2.)
    svals = np.linalg.svd(thetas, compute_uv=False)
    return svals


@cite('freud2016')
def neighborhood_range_angle_singvals(box, positions, neigh_min, neigh_max, rmax_guess=2.):
    """Construct a matrix of pairwise angles between `(rk - ri)` and `(rj -
    ri)` for all neighbors j and k(==j) of each particle i, for a range
    of neighborhood sizes from neigh_min to neigh_max
    (inclusive). Returns the singular values of this matrix to fix
    permutation invariance.
    """
    result = []

    for neighbors in range(neigh_min, neigh_max + 1):
        result.append(neighborhood_angle_singvals(box, positions, neighbors, rmax_guess))

    return np.hstack(result)


@cite('freud2016')
def neighborhood_angle_sorted(box, positions, neighbors, rmax_guess=2.):
    """Construct a matrix of pairwise angles between `(rk - ri)` and `(rj -
    ri)` for all neighbors j and k(==j) of each particle i, for a
    particular number of neighbors. Returns the sorted values of
    this matrix to fix permutation invariance.
    """
    thetas = _get_neighborhood_angle_matrix(box, positions, neighbors, rmax_guess=2.)
    thetas = thetas.reshape((thetas.shape[0], -1))
    np.sort(thetas)
    return thetas
