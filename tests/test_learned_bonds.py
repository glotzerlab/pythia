import keras
import numpy as np
import pythia
import pythia.learned
import rowan
import tensorflow as tf
import unittest


class TestLearnedBonds(unittest.TestCase):
    def setUp(self):
        np.random.seed(12)
        tf.set_random_seed(13)

    def test_find_max_distance(self):
        num_training = 1024
        num_neighbors = 4

        train_points = np.random.uniform(-1, 1, (num_training, num_neighbors, 3))
        validation_points = np.random.uniform(-1, 1, (num_training, num_neighbors, 3))

        train_classes = np.argmax(np.linalg.norm(train_points, axis=-1), axis=-1)
        validation_classes = np.argmax(np.linalg.norm(validation_points, axis=-1), axis=-1)

        validation_data = (validation_points,
                           keras.utils.np_utils.to_categorical(validation_classes, num_neighbors))

        model = keras.models.Sequential()
        model.add(pythia.learned.bonds.InertiaRotation(8, input_shape=train_points.shape[1:]))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(num_neighbors, activation='softmax'))
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        train_history = model.fit(
            train_points,
            keras.utils.np_utils.to_categorical(train_classes, num_neighbors),
            validation_data=validation_data, epochs=200, verbose=0)

        # we are ending quite early, but 50% accuracy looks to be
        # pretty easy to achieve
        self.assertGreater(train_history.history['val_acc'][-1], .5)


if __name__ == '__main__':
    unittest.main()
