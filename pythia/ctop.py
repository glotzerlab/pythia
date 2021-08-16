"""
This module computes Minkowski Structure Metrics of the system.
"""

import numpy as np
import freud

from .internal import cite


@cite('mickel2013')
def minkowski_structure_metrics(
    box, positions, qls=None, wls=None, qls_ave=None, wls_ave=None
):
    """Compute a set of Minkowski Structure Metrics for each particle.

    :param box: Periodic box object.
    :param positions: Particle positions.
    :param qls: List of l-values used to compute :math:`q'_l`.
    :param wls: List of l-values used to compute :math:`w'_l`.
    :param qls_ave: List of l-values used to compute :math:`\overline{q}'_l`.
    :param wls_ave: List of l-values used to compute :math:`\overline{w}'_l`.
    """

    if qls is None:
        qls = []
    if wls is None:
        wls = []
    if qls_ave is None:
        qls_ave = []
    if wls_ave is None:
        wls_ave = []
    # Sanitize input arrays so they are list[int]
    qls = np.asarray(qls).astype(int).tolist()
    wls = np.asarray(wls).astype(int).tolist()
    qls_ave = np.asarray(qls_ave).astype(int).tolist()
    wls_ave = np.asarray(wls_ave).astype(int).tolist()
    system = (box, positions)
    descriptors = []
    if len(qls) or len(wls) or len(qls_ave) or len(wls_ave):
        voro = freud.locality.Voronoi()
        voro.compute(system)
    if len(qls):
        st = freud.order.Steinhardt(l=qls, weighted=True)
        st.compute(system, neighbors=voro.nlist)
        descriptors.append(st.particle_order)
    if len(wls):
        st = freud.order.Steinhardt(l=wls, wl=True, weighted=True, wl_normalize=True)
        st.compute(system, neighbors=voro.nlist)
        descriptors.append(st.particle_order)
    if len(qls_ave):
        st = freud.order.Steinhardt(l=qls_ave, average=True, weighted=True)
        st.compute(system, neighbors=voro.nlist)
        descriptors.append(st.particle_order)
    if len(wls_ave):
        st = freud.order.Steinhardt(
            l=wls_ave, wl=True, average=True, weighted=True, wl_normalize=True
        )
        st.compute(system, neighbors=voro.nlist)
        descriptors.append(st.particle_order)
    for i in range(len(descriptors)):
        if descriptors[i].ndim == 1:
            descriptors[i] = descriptors[i].reshape(-1, 1)
    descriptors = np.concatenate(descriptors, axis=1)
    return descriptors
