import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


@tf.custom_gradient
def _custom_eigvecsh(x):
    # increase the stability of the eigh calculation by removing nans
    # and infs (nans occur when there are two identical eigenvalues,
    # for example) and clipping the gradient magnitude
    (evals, evecs) = tf.linalg.eigh(x)
    def grad(dvecs):
        dvecs = tf.where(tf.is_finite(dvecs), dvecs, tf.zeros_like(dvecs))
        dvecs = K.clip(dvecs, -1, 1)
        return dvecs

    return evecs, grad


@tf.custom_gradient
def _ignore_nan_gradient(x):
    result = K.identity(x)
    def grad(dy):
        dy = tf.where(tf.is_finite(dy), dy, tf.zeros_like(dy))
        return dy

    return result, grad


def _diagonalize(xyz, mass):
    rsq = K.expand_dims(K.sum(xyz**2, axis=-1, keepdims=True), -1)
    # xyz::(..., num_neighbors, 3)
    # f1, f2::(..., num_neighbors, 3, 3)
    f1 = K.eye(3)*rsq
    f2 = K.expand_dims(xyz, -2)*K.expand_dims(xyz, -1)
    # mass::(..., num_neighbors)
    expanded_mass = K.expand_dims(K.expand_dims(mass, -1), -1)
    # I_s::(..., 3, 3)
    I_s = K.sum((f1 - f2)*expanded_mass, -3)

    # rotations::(..., 3, 3)
    rotations = _custom_eigvecsh(I_s)

    # swap z for -z when an inversion occurs
    det_sign = tf.linalg.det(rotations)
    inversions = K.stack(
        [K.ones_like(det_sign), K.ones_like(det_sign), det_sign], axis=-1)
    rotations = rotations*K.expand_dims(inversions, -2)

    rotated_xyz = K.sum(K.expand_dims(xyz, -1)*K.expand_dims(rotations, -3), -2)

    return rotated_xyz, I_s, rotations


class InertiaRotation(keras.layers.Layer):
    """Generate rotation-invariant point clouds by orienting via principal axes of inertia

    `InertiaRotation` takes an array of neighborhood points
    (shape `(..., num_neighbors, 3)`) and outputs one or more copies
    which have been rotated according to the principal axes of inertia
    of the neighborhood. It does this using masses that can be varied
    for each point and each rotation.

    For an input of shape `(..., num_neighbors, 3)`,
    `InertiaRotation` produces an output of shape `(...,
    num_rotations, num_neighbors, 3)`.

    Before computing the inertia tensor, points can optionally be
    centered via the `center` argument.  A value of `True` centers the
    points as if all masses were 1, a value of `"com"` centers the
    points using the learned masses, and a value of `False` (the
    default) does not center at all.

    :param num_rotations: number of rotations to produce
    :param initial_mass_variance: Variance of the initial mass distribution (mean 1)
    :param center: Center the mass points before computing the inertia tensor (see description above)
    """
    def __init__(self, num_rotations=1, initial_mass_variance=.25,
                 center=False, **kwargs):
        self.num_rotations = num_rotations
        self.initial_mass_variance = float(initial_mass_variance)

        if center not in (False, True, 'com'):
            msg = ('Center argument {} must be a bool or "com" (to '
                'center using the mass stored in this layer)'.format(center))
            raise ArgumentError(msg)

        self.center = center

        super(InertiaRotation, self).__init__(**kwargs)

    def build(self, input_shape):
        mass_shape = [self.num_rotations] + list(input_shape[-2:-1])

        self.mass = self.add_weight(
            'mass', mass_shape,
            initializer=keras.initializers.RandomNormal(1., self.initial_mass_variance),
            constraint=keras.constraints.NonNeg())
        self.mass = self.mass/K.sum(self.mass, -1, keepdims=True)
        self.mass = _ignore_nan_gradient(self.mass)

        super(InertiaRotation, self).build(input_shape)

    def call(self, neighborhood_xyz):
        # neighborhood_xyz: (..., num_neighbors, 3) -> (..., num_rotations, num_neighbors, 3)
        repeats = np.ones(len(neighborhood_xyz.shape) + 1)
        repeats[-3] = self.num_rotations
        neighborhood_xyz = K.expand_dims(neighborhood_xyz, -3)

        if self.center == 'com':
            # mass for each neighborhood is already normalized to sum to 1
            center_of_mass = K.sum(
                neighborhood_xyz*K.expand_dims(self.mass, -1), -2, keepdims=True)
            neighborhood_xyz = neighborhood_xyz - center_of_mass
        elif self.center:
            neighborhood_xyz = neighborhood_xyz - K.mean(neighborhood_xyz, -2, keepdims=True)

        (self.diagonal_xyz, self.inertia_tensors, self.rotations) = \
            _diagonalize(neighborhood_xyz, self.mass)
        return self.diagonal_xyz

    def compute_output_shape(self, input_shape):
        # (..., num_neighbors, 3)->(..., num_rotations, num_neighbors, 3)
        shape = list(input_shape)
        shape.insert(-2, self.num_rotations)
        return tuple(shape)

    def get_config(self):
        config = super().get_config()
        config.update(dict(num_rotations=self.num_rotations,
                           initial_mass_variance=self.initial_mass_variance,
                           center=self.center))
        return config
