"""
This module computes descriptors based on the Voronoi tessellation
of the system.
"""

import numpy as np
import freud

import scipy as sp
import scipy.spatial

from .internal import cite


def _angle_histogram_3d(vertices, bins=16, area_weight_mode='product'):
    hull = sp.spatial.ConvexHull(vertices)
    dot_products = np.zeros((hull.equations.shape[0], hull.equations.shape[0]), dtype=np.float32)

    # equations are already normalized for the (x, y, z) components
    dot_products[:, :] = np.sum(hull.equations[:, np.newaxis, :3] *
                                hull.equations[np.newaxis, :, :3], axis=-1)
    angles = np.arccos(dot_products)

    facet_vertices = hull.points[hull.simplices]
    facet_vertices -= facet_vertices[:, 0, np.newaxis, :]
    facet_AB = facet_vertices[:, 1, :]
    facet_AC = facet_vertices[:, 2, :]

    mag_AB = np.sqrt(np.sum(facet_AB**2, axis=-1, keepdims=True))
    mag_AC = np.sqrt(np.sum(facet_AC**2, axis=-1, keepdims=True))
    normalized_AB = facet_AB/mag_AB
    normalized_AC = facet_AC/mag_AC
    facet_thetas = np.arccos(np.sum(normalized_AB*normalized_AC, axis=-1))

    facet_areas = .5*mag_AB.reshape((-1,))*mag_AC.reshape((-1,))*np.sin(facet_thetas)

    if area_weight_mode in ('+', 'sum'):
        area_weights = facet_areas[:, np.newaxis] + facet_areas[np.newaxis, :]
    elif area_weight_mode in ('*', 'product'):
        area_weights = facet_areas[:, np.newaxis]*facet_areas[np.newaxis, :]
    else:
        raise NotImplementedError('Unknown area_weight_mode: "{}"'.format(area_weight_mode))
    np.fill_diagonal(area_weights, 0)

    bin_targets = np.round(angles/np.pi*(bins - 1)).astype(np.int32)
    bin_targets = bin_targets.reshape((-1,))
    area_weights = area_weights.reshape((-1,))

    result = np.zeros((bins,), dtype=np.float32)

    for (target, weight) in zip(bin_targets, area_weights):
        result[target] += weight

    return result/np.sum(area_weights)


@cite('freud2016')
def angle_histogram(box, positions, bins, buffer_distance=None, area_weight_mode='product'):
    """Compute the area-weighted (a_i + a_j) angle histogram of all pairs
    of faces for the voronoi polyhedron of each particle. Sums the areas
    into the given number of bins (from 0 to pi).

    :param bins: Number of bins to use for the histogram
    :param buffer_distance: Distance to copy parts of the simulation box for periodic boundary conditions in the voronoi diagram computation
    :param area_weight_mode: Whether the weight for each pair of faces should be the sum ('sum') or product ('product') of the face areas
    """  # noqa E501

    # The buffer distance is used to produce image for the periodic boundary condition,
    # to avoid corner cases like particles are only in one quadrant, we need to take the max
    # value of the box dimension as the replication buffer distance.
    if buffer_distance is None:
        buffer_distance = max(box.Lx, box.Ly, box.Lz)

    fbox = freud.box.Box.from_box(box)
    voronoi = freud.voronoi.Voronoi(fbox, buff=buffer_distance)
    voronoi.compute(positions)

    all_polyhedra = voronoi.polytopes
    polyhedra = [all_polyhedra[i] for i in range(len(positions))]

    result = np.zeros((len(positions), bins), dtype=np.float32)
    for (i, verts) in enumerate(polyhedra):
        result[i] = _angle_histogram_3d(verts, bins, area_weight_mode)

    return result
