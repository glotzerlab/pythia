"""
This module computes descriptors based on combinations of
spherical harmonics applied to nearest-neighbor bonds.
"""

from collections import defaultdict
import itertools
import logging
import numpy as np
import freud

from .internal import assert_installed, cite

logger = logging.getLogger(__name__)


def _nlist_helper(fbox, positions, neighbors, rmax_guess=2., exclude_ii=None):
    if isinstance(neighbors, int):
        nneigh = freud.locality.NearestNeighbors(rmax_guess, neighbors)
        nneigh.compute(fbox, positions, positions, exclude_ii)
        neighbors = nneigh.nlist
    elif isinstance(neighbors, float):
        lc = freud.locality.LinkCell(fbox, neighbors)
        lc.compute(fbox, positions, positions, exclude_ii)
        neighbors = lc.nlist

    return neighbors


@cite('freud2016', 'spellings2018')
def neighbor_average(box, positions, neigh_min=4, neigh_max=4, lmax=4,
                     negative_m=True, reference_frame='neighborhood',
                     orientations=None, rmax_guess=1., noise_samples=0,
                     noise_magnitude=0, nlist=None):
    """Compute the neighbor-averaged spherical harmonics over the
    nearest-neighbor bonds of a set of particles. Returns the raw
    (complex) spherical harmonic values.

    :param neigh_min: Minimum number of neighbor environment sizes to consider
    :param neigh_max: Maximum number of neighbor environment sizes to consider (inclusive)
    :param lmax: Maximum spherical harmonic degree l
    :param negative_m: Include negative m spherical harmonics in the output array?
    :param reference_frame: 'neighborhood': use diagonal inertia tensor reference frame; 'particle_local': use the given orientations array; 'global': do not rotate
    :param orientations: Per-particle orientations, only used when reference_frame == 'particle_local'
    :param rmax_guess: Initial guess of the distance to find `neigh_max` nearest neighbors. Only affects algorithm speed.
    :param noise_samples: Number of random noisy samples of positions to average the result over (disabled if 0)
    :param noise_magnitude: Magnitude of (normally-distributed) noise to apply to noise_samples different positions (disabled if `noise_samples == 0`)
    :param nlist: Freud neighbor list object to use (`None` to compute for neighbors up to `neigh_max`)
    """  # noqa E501
    freud_box = freud.box.Box.from_box(box)

    if noise_samples:
        accumulation = 0
        for _ in range(noise_samples):
            noise = np.random.normal(0, noise_magnitude, positions.shape)
            noisy_positions = positions + noise
            freud_box.wrap(noisy_positions)
            noisy_descriptors = neighbor_average(
                box, noisy_positions, neigh_min, neigh_max, lmax, negative_m,
                reference_frame, orientations, rmax_guess, 0, 0)

            accumulation += noisy_descriptors

        accumulation /= noise_samples
        return accumulation

    if orientations is None and reference_frame == 'particle_local':
        logger.error('reference_frame="particle_local" was given for '
                     'neighbor_average, but orientations were not given')
        orientations = np.zeros((positions.shape[0], 4), dtype=np.float32)
        orientations[:, 0] = 1

    result = []
    comp = freud.environment.LocalDescriptors(neigh_max, lmax, rmax_guess, negative_m)
    if nlist is None:
        nn = freud.locality.NearestNeighbors(rmax_guess, neigh_max)
        nn.compute(freud_box, positions, positions)
        nlist = nn.nlist

    neighbor_counts = nlist.neighbor_counts
    if np.any(neighbor_counts < neigh_max):
        indices = np.where(neighbor_counts < neigh_max)[0]
        logger.warning('{} particles have too few neighbors'.format(len(indices)))

    for nNeigh in range(neigh_min, neigh_max + 1):
        # sphs::(Nbond, Nsph)
        comp.compute(freud_box, nNeigh, positions, positions, orientations, nlist=nlist)
        sphs = comp.sph

        # average over neighbors
        sphs = np.add.reduceat(sphs, nlist.segments)
        sphs /= np.clip(neighbor_counts, 1, nNeigh)[:, np.newaxis]
        result.append(sphs)

    return np.hstack(result)


@cite('freud2016', 'spellings2018')
def abs_neighbor_average(box, positions, neigh_min=4, neigh_max=4, lmax=4,
                         negative_m=True, reference_frame='neighborhood',
                         orientations=None, rmax_guess=1., noise_samples=0,
                         noise_magnitude=0, nlist=None):
    """Compute the neighbor-averaged spherical harmonics over the
    nearest-neighbor bonds of a set of particles. Returns the absolute
    value of the (complex) spherical harmonics

    :param neigh_min: Minimum number of neighbor environment sizes to consider
    :param neigh_max: Maximum number of neighbor environment sizes to consider (inclusive)
    :param lmax: Maximum spherical harmonic degree l
    :param negative_m: Include negative m spherical harmonics in the output array?
    :param reference_frame: 'neighborhood': use diagonal inertia tensor reference frame; 'particle_local': use the given orientations array; 'global': do not rotate
    :param orientations: Per-particle orientations, only used when reference_frame == 'particle_local'
    :param rmax_guess: Initial guess of the distance to find `neigh_max` nearest neighbors. Only affects algorithm speed.
    :param noise_samples: Number of random noisy samples of positions to average the result over (disabled if 0)
    :param noise_magnitude: Magnitude of (normally-distributed) noise to apply to noise_samples different positions (disabled if `noise_samples == 0`)
    :param nlist: Freud neighbor list object to use (`None` to compute for neighbors up to `neigh_max`)
    """  # noqa E501
    return np.abs(neighbor_average(
        box, positions, neigh_min, neigh_max, lmax, negative_m,
        reference_frame, orientations, rmax_guess, noise_samples, noise_magnitude,
        nlist))


@cite('freud2016', 'spellings2018')
def system_average(box, positions, neigh_min=4, neigh_max=4, lmax=4,
                   negative_m=True, reference_frame='neighborhood',
                   orientations=None, rmax_guess=1., noise_samples=0,
                   noise_magnitude=0, nlist=None):
    """Compute the global-averaged spherical harmonics over the
    nearest-neighbor bonds of a set of particles. Returns the raw
    (complex) spherical harmonic values.

    :param neigh_min: Minimum number of neighbor environment sizes to consider
    :param neigh_max: Maximum number of neighbor environment sizes to consider (inclusive)
    :param lmax: Maximum spherical harmonic degree l
    :param negative_m: Include negative m spherical harmonics in the output array?
    :param reference_frame: 'neighborhood': use diagonal inertia tensor reference frame; 'particle_local': use the given orientations array; 'global': do not rotate
    :param orientations: Per-particle orientations, only used when reference_frame == 'particle_local'
    :param rmax_guess: Initial guess of the distance to find `neigh_max` nearest neighbors. Only affects algorithm speed.
    :param noise_samples: Number of random noisy samples of positions to average the result over (disabled if 0)
    :param noise_magnitude: Magnitude of (normally-distributed) noise to apply to noise_samples different positions (disabled if `noise_samples == 0`)
    :param nlist: Freud neighbor list object to use (`None` to compute for neighbors up to `neigh_max`)
    """  # noqa E501
    return np.mean(neighbor_average(
        box, positions, neigh_min, neigh_max, lmax, negative_m,
        reference_frame, orientations, rmax_guess, noise_samples, noise_magnitude,
        nlist), axis=0)


@cite('freud2016', 'spellings2018')
def abs_system_average(box, positions, neigh_min=4, neigh_max=4, lmax=4,
                       negative_m=True, reference_frame='neighborhood',
                       orientations=None, rmax_guess=1., noise_samples=0,
                       noise_magnitude=0, nlist=None):
    """Compute the global-averaged spherical harmonics over the
    nearest-neighbor bonds of a set of particles. Returns the absolute
    value of the (complex) spherical harmonics

    :param neigh_min: Minimum number of neighbor environment sizes to consider
    :param neigh_max: Maximum number of neighbor environment sizes to consider (inclusive)
    :param lmax: Maximum spherical harmonic degree l
    :param negative_m: Include negative m spherical harmonics in the output array?
    :param reference_frame: 'neighborhood': use diagonal inertia tensor reference frame; 'particle_local': use the given orientations array; 'global': do not rotate
    :param orientations: Per-particle orientations, only used when reference_frame == 'particle_local'
    :param rmax_guess: Initial guess of the distance to find `neigh_max` nearest neighbors. Only affects algorithm speed.
    :param noise_samples: Number of random noisy samples of positions to average the result over (disabled if 0)
    :param noise_magnitude: Magnitude of (normally-distributed) noise to apply to noise_samples different positions (disabled if `noise_samples == 0`)
    :param nlist: Freud neighbor list object to use (`None` to compute for neighbors up to `neigh_max`)
    """  # noqa E501
    return np.abs(system_average(
        box, positions, neigh_min, neigh_max, lmax, negative_m,
        reference_frame, orientations, rmax_guess, noise_samples, noise_magnitude,
        nlist))


@cite('freud2016')
def steinhardt_q(box, positions, neighbors=12, lmax=6, rmax_guess=2.):
    """Compute a vector of per-particle Steinhardt order parameters.

    :param neighbors: Number of neighbors (int) or maximum distance to find neighbors within (float)
    :param lmax: Maximum spherical harmonic degree l
    :param rmax_guess: Initial guess of the distance to find nearest neighbors, if appropriate. Only affects algorithm speed.
    """  # noqa E501
    box = freud.box.Box.from_box(box)
    neighbors = _nlist_helper(box, positions, neighbors, rmax_guess)

    result = []
    for l in range(2, lmax + 1, 2):
        compute = freud.order.LocalQl(box, rmax_guess, l)
        compute.compute(positions, neighbors)
        op = compute.Ql
        result.append(op.copy())

    result = np.array(result, dtype=np.float32).T
    return result


class _clebsch_gordan_cache(object):
    _cache = {}

    @classmethod
    def get(cls, l1, l2, l3, m1, m2, m3):
        sympy = assert_installed('sympy')
        assert_installed('sympy.physics.wigner')
        key = (l1, l2, l3, m1, m2, m3)

        if key not in cls._cache:
            cls._cache[key] = float(sympy.physics.wigner.clebsch_gordan(*key))

        return cls._cache[key]


@cite('kondor2007', 'freud2016')
def bispectrum(box, positions, neighbors, lmax, rmax_guess=2.):
    """Computes bispectrum invariants of particle local
    environments. These are rotationally-invariant descriptions
    similar to a power spectrum of the spherical harmonics
    (i.e. steinhardt order parameters), but retaining more
    information.

    :param neighbors: number of nearest-neighbors to consider for local environments
    :param lmax: maximum spherical harmonic degree to consider. O(lmax**3) descriptors will be generated.
    """  # noqa E501
    fsph = assert_installed('fsph')

    box = freud.box.Box.from_box(box)
    nlist = _nlist_helper(box, positions, neighbors, rmax_guess)

    rijs = positions[nlist.index_j] - positions[nlist.index_i]
    box.wrap(rijs)

    phi = np.arccos(rijs[..., 2]/np.sqrt(np.sum(rijs**2, axis=-1)))
    theta = np.arctan2(rijs[..., 1], rijs[..., 0])

    sphs = fsph.pointwise_sph(phi, theta, lmax, negative_m=True)
    sphs = np.add.reduceat(sphs, nlist.segments)/nlist.neighbor_counts[:, np.newaxis]
    sphs[np.isnan(sphs)] = 0
    lm_columns = {(l, m): i for (i, (l, m)) in enumerate(fsph.get_LMs(lmax, negative_m=True))}

    for (_, m), i in lm_columns.items():
        if m > 0 and m % 2:
            sphs[:, i] *= -1

    result = defaultdict(lambda: 0)
    for (l1, l2, l) in itertools.product(range(lmax + 1), range(lmax + 1), range(lmax + 1)):
        result_key = (l1, l2, l)

        for m in range(-l, l + 1):
            left = sphs[:, lm_columns[(l, m)]]

            right = 0 + 0j

            nonzero = False
            m1_min = max(-l1, m - l2)
            m1_max = min(l1, m + l2)
            for m1 in range(m1_min, m1_max + 1):
                term = _clebsch_gordan_cache.get(l1, l2, l, m1, m - m1, m)

                if term == 0:
                    continue
                else:
                    nonzero = True

                term *= np.conj(sphs[:, lm_columns[(l1, m1)]])
                term *= np.conj(sphs[:, lm_columns[(l2, m - m1)]])

                right += term

            if nonzero:
                result[result_key] += left*right

    result_columns = [result[key] for key in sorted(result)]
    result = np.array(result_columns, dtype=np.complex128).T
    result = np.ascontiguousarray(result).view(np.float64).reshape((positions.shape[0], -1))

    return result
