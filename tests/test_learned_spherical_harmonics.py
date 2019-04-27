import keras
import numpy as np
import pythia
import pythia.learned
import rowan
import tensorflow as tf
import unittest


class TestLearnedSphericalHarmonics(unittest.TestCase):
    def setUp(self):
        np.random.seed(12)
        tf.set_random_seed(13)

    def test_basic_point_clouds(self):
        classes = [
            [(-1., 0, 0), (1, 0, 0), (-2, 0, 0), (2, 0, 0)], # line of points
            [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)], # square
        ]
        num_classes = len(classes)
        num_data = 128

        train_ins = np.tile(classes, (num_data, 1, 1))
        train_ins += np.random.normal(scale=.1, size=train_ins.shape)

        # sort by R to learn permutation invariance
        rsqs = np.linalg.norm(train_ins, axis=-1)
        for (neighborhood, rsq) in zip(train_ins, rsqs):
            neighborhood[:] = neighborhood[np.argsort(rsq)]

        train_ins, test_ins = train_ins[:train_ins.shape[0]//2], train_ins[train_ins.shape[0]//2:]
        train_outs = np.tile(np.arange(num_classes), num_data//2)

        model = keras.models.Sequential()
        model.add(pythia.learned.bonds.InertiaRotation(2, input_shape=train_ins.shape[1:]))
        model.add(pythia.learned.spherical_harmonics.SphericalHarmonics(2))
        model.add(pythia.learned.spherical_harmonics.NeighborAverage())
        model.add(pythia.learned.spherical_harmonics.ComplexProjection(4, 'abs'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        net_train_outs = keras.utils.np_utils.to_categorical(train_outs, num_classes)

        train_history = model.fit(
            train_ins, net_train_outs, validation_data=(test_ins, net_train_outs),
            epochs=200, verbose=0)

        self.assertGreater(train_history.history['val_acc'][-1], .75)

    def test_chiral_point_clouds(self):
        classes = [
            [(0., 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)], # a shape...
            [(0, 0, 0), (-1, 0, 0), (0, -2, 0), (0, 0, -3)], # and an inversion of that shape
        ]
        num_classes = len(classes)
        num_data = 128

        train_ins = np.tile(classes, (num_data, 1, 1))
        train_ins += np.random.normal(scale=.1, size=train_ins.shape)

        train_ins, test_ins = train_ins[:train_ins.shape[0]//2], train_ins[train_ins.shape[0]//2:]
        train_outs = np.tile(np.arange(num_classes), num_data//2)

        quats = rowan.random.rand(test_ins.shape[0])
        train_ins = rowan.rotate(quats[:, np.newaxis], train_ins)
        test_ins = rowan.rotate(quats[:, np.newaxis], test_ins)

        model = keras.models.Sequential()
        model.add(pythia.learned.bonds.InertiaRotation(
            2, center='com', input_shape=train_ins.shape[1:]))
        model.add(pythia.learned.spherical_harmonics.SphericalHarmonics(4))
        model.add(pythia.learned.spherical_harmonics.NeighborAverage())
        model.add(pythia.learned.spherical_harmonics.ComplexProjection(4, 'abs,angle'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        net_train_outs = keras.utils.np_utils.to_categorical(train_outs, num_classes)

        train_history = model.fit(
            train_ins, net_train_outs, validation_data=(test_ins, net_train_outs),
            epochs=100, verbose=0)

        self.assertGreater(np.max(train_history.history['val_acc']), .75)

    def test_rotational_invariance(self):
        classes = [
            [(-1., 0, 0), (1., 0, 0)],
            [(0, 1, 0), (0, -1, 0)]
        ]
        num_classes = len(classes)
        num_data = 128

        train_ins = np.tile(classes, (num_data, 1, 1))
        train_ins += np.random.normal(scale=.1, size=train_ins.shape)

        # sort by R to learn permutation invariance
        rsqs = np.linalg.norm(train_ins, axis=-1)
        for (neighborhood, rsq) in zip(train_ins, rsqs):
            neighborhood[:] = neighborhood[np.argsort(rsq)]

        train_ins, test_ins = train_ins[:train_ins.shape[0]//2], train_ins[train_ins.shape[0]//2:]
        train_outs = np.tile(np.arange(num_classes), num_data//2)

        quats = rowan.random.rand(test_ins.shape[0])
        test_ins = rowan.rotate(quats[:, np.newaxis], test_ins)

        model = keras.models.Sequential()
        model.add(pythia.learned.bonds.InertiaRotation(
            2, input_shape=train_ins.shape[1:]))
        model.add(pythia.learned.spherical_harmonics.SphericalHarmonics(2))
        model.add(pythia.learned.spherical_harmonics.NeighborAverage())
        model.add(pythia.learned.spherical_harmonics.ComplexProjection(4, 'abs'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        net_train_outs = keras.utils.np_utils.to_categorical(train_outs, num_classes)

        train_history = model.fit(
            train_ins, net_train_outs, validation_data=(test_ins, net_train_outs),
            epochs=200, verbose=0)

        self.assertLess(np.mean(train_history.history['val_acc']), .55)


if __name__ == '__main__':
    unittest.main()
