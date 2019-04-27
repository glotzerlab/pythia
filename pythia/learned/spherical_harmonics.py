import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

import fsph
import fsph.tf_ops


def _xyz_to_phitheta(xyz):
    r = tf.linalg.norm(xyz, axis=-1)
    phi = tf.acos(K.clip(xyz[..., 2]/r, -1, 1))
    theta = tf.atan2(xyz[..., 1], xyz[..., 0])

    phitheta = K.stack([phi, theta], -1)
    return phitheta


class SphericalHarmonics(keras.layers.Layer):
    """Compute the (complex) spherical harmonic decomposition given a set of cartesian coordinates

    For an input of shape `(..., 3)`, `SphericalHarmonics` will
    produce an output of shape `(..., num_sphs)`, where `num_sphs` is
    the number of spherical harmonics produced given the `lmax` and
    `negative_m` arguments.

    :param lmax: maximum spherical harmonic degree to compute
    :param negative_m: If True, consider m=-l to m=l for each spherical harmonic l; otherwise, consider m=0 to m=l
    """
    def __init__(self, lmax, negative_m=False, **kwargs):
        self.lmax = int(lmax)
        self.negative_m = bool(negative_m)

        self.num_sphs = len(fsph.get_LMs(self.lmax, self.negative_m))

        super(SphericalHarmonics, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] == 3, 'SphericalHarmonics must take an array of (x, y, z) values'
        super(SphericalHarmonics, self).build(input_shape)

    def call(self, xyz):
        self.diagonal_phitheta = _xyz_to_phitheta(xyz)
        self.spherical_harmonics = fsph.tf_ops.spherical_harmonic_series(
            self.diagonal_phitheta, self.lmax, self.negative_m)
        return self.spherical_harmonics

    def compute_output_shape(self, input_shape):
        # (..., 3) -> (..., num_sphs)
        shape = list(input_shape)
        shape[-1] = self.num_sphs
        return tuple(shape)

    def get_config(self):
        config = super().get_config()
        config.update(dict(lmax=self.lmax,
                           negative_m=self.negative_m))
        return config


class NeighborAverage(keras.layers.Layer):
    """Compute a weighted average an array of complex-valued spherical harmonics over all points in a neighborhood

    Given an input of shape `(..., num_rotations, num_neighbors,
    num_sphs)`, `NeighborAverage` will produce an output of shape
    `(..., num_rotations, num_sphs)`. Each neighbor from each rotation
    is assigned a learnable weight to contribute to the sum.
    """
    def build(self, input_shape):
        # (rotations, neighbors)
        shape = (input_shape[-3], input_shape[-2],)
        weight_scale = 1.0/input_shape[-2]
        self.neighbor_weights = self.add_weight(
            'neighbor_weights', shape,
            initializer=keras.initializers.RandomUniform(-weight_scale, weight_scale))
        self.neighbor_weights = K.expand_dims(self.neighbor_weights, -1)

        for _ in range(self.neighbor_weights.shape.ndims, len(input_shape)):
            self.neighbor_weights = K.expand_dims(self.neighbor_weights, 0)

        super(NeighborAverage, self).build(input_shape)

    def call(self, inputs):
        self.neighbor_sum = K.sum(inputs*K.cast(self.neighbor_weights, tf.complex64), -2)
        return self.neighbor_sum

    def compute_output_shape(self, input_shape):
        # (..., num_neighbors, num_sphs) -> (..., num_sphs)
        shape = list(input_shape)
        shape.pop(-2)
        return tuple(shape)


class ComplexProjection(keras.layers.Layer):
    """Compute one or more linear projections of a complex-valued function

    Given an input of shape `(..., num_rotations, num_sphs)`,
    `ComplexProjection` produces an output of shape `(..,
    num_rotations, num_projections)`.

    Outputs are converted to real numbers by taking the absolute value
    of the projection output by default, but the real or imaginary
    components can also be taken instead.

    :param num_projections: Number of projections (i.e. number of neurons) to create for each rotation
    :param conversion: Method to make the projection output real: 'abs' (absolute value), 'angle' (complex phase), 'real' (real component), 'imag' (imaginary component), or a comma-separated list of these values (i.e. 'real,imag')
    """
    def __init__(self, num_projections=1, conversion='abs', **kwargs):
        self.num_projections = int(num_projections)
        self.conversion = conversion

        super(ComplexProjection, self).__init__(**kwargs)

    def build(self, input_shape):
        # (rotations, spherical_harmonics, projections)
        shape = (input_shape[-2], input_shape[-1], self.num_projections)
        weight_scale = np.sqrt(6.0/(shape[-2] + shape[-1]))
        self.projection = self.add_weight(
            'projection', shape,
            initializer=keras.initializers.RandomUniform(-weight_scale, weight_scale))

        for _ in range(len(shape), len(input_shape)):
            self.projection = K.expand_dims(self.projection, 0)

        super(ComplexProjection, self).build(input_shape)

    def call(self, inputs):
        # inputs::(..., rotations, spherical_harmonics)
        self.sph_projected = K.sum(
            K.expand_dims(inputs, -1)*K.cast(self.projection, tf.complex64), -2)

        conversions = self.conversion.split(',')
        result = []
        for conversion in conversions:
            conversion = conversion.strip()
            if conversion == 'abs':
                result.append(K.abs(self.sph_projected))
            elif conversion == 'angle':
                result.append(tf.angle(self.sph_projected))
            elif conversion == 'real':
                result.append(tf.real(self.sph_projected))
            elif conversion == 'imag':
                result.append(tf.imag(self.sph_projected))
            else:
                raise NotImplementedError('Unknown conversion {}'.format(conversion))

        if len(conversions) > 1:
            self.projected = K.stack(result, -1)
        else:
            self.projected = result[0]

        return self.projected

    def compute_output_shape(self, input_shape):
        # (..., num_sphs) -> (..., num_projections)
        shape = list(input_shape)
        shape[-1] = self.num_projections*len(self.conversion.split(','))
        return tuple(shape)

    def get_config(self):
        config = super().get_config()
        config.update(dict(num_projections=self.num_projections,
                           conversion=self.conversion))
        return config
