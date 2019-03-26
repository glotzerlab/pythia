import numpy as np
import freud
import pythia
import unittest


class TestBonds(unittest.TestCase):
    def test_neighborhood_angle_singvals(self):
        N = 1000
        Nneigh = 12

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        descriptors = pythia.bonds.neighborhood_angle_singvals(
            box, positions, Nneigh)

        self.assertEqual(descriptors.shape, (N, Nneigh))

    def test_get_neighborhood_distance_matrix(self):
        Nneigh = 6
        box = freud.box.Box.cube(10)
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0],
             [0, 0, 1], [0, 1, 1], [0, 2, 1],
             [1, 0, 1], [1, 1, 1], [1, 2, 1],
             [2, 0, 1], [2, 1, 1], [2, 2, 1],
             [0, 0, 2], [0, 1, 2], [0, 2, 2],
             [1, 0, 2], [1, 1, 2], [1, 2, 2],
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float32)

        matrix = pythia.bonds._get_neighborhood_distance_matrix(
            box, positions, Nneigh, rmax_guess=2.)

        # Particle 13 at [1, 1, 1] would have closest neighbors at
        # [1 +- 1, 1 +- 1, 1 +- 1], so the unnormalized distance matrix would be
        # consisting of only sqrt(2) and 2, while normalized one would only have
        # 1 and sqrt(2).

        idx_111 = 13
        self.assertAlmostEquals(matrix[idx_111, 0, 0], 0)

        # By definition of the noramlization.
        self.assertAlmostEquals(matrix[idx_111, 0, 1], 1)

        # By calculating distance and normalize by sqrt(2).
        np.testing.assert_almost_equal(matrix[idx_111, 0, 1:],
                                       [1, 1, 1, 1, np.sqrt(2)])

    def test_get_neighborhood_angle_matrix(self):
        Nneigh = 6
        box = freud.box.Box.cube(10)
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0],
             [0, 0, 1], [0, 1, 1], [0, 2, 1],
             [1, 0, 1], [1, 1, 1], [1, 2, 1],
             [2, 0, 1], [2, 1, 1], [2, 2, 1],
             [0, 0, 2], [0, 1, 2], [0, 2, 2],
             [1, 0, 2], [1, 1, 2], [1, 2, 2],
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float32)

        matrix = pythia.bonds._get_neighborhood_angle_matrix(
            box, positions, Nneigh, rmax_guess=2.)

        # Particle 13 at [1, 1, 1] would have closest neighbors at
        # [1 +- 1, 1 +- 1, 1 +- 1], so the unnormalized distance matrix would be
        # consisting of only sqrt(2) and 2, while normalized one would only have
        # 1 and sqrt(2).

        idx_111 = 13
        self.assertAlmostEquals(matrix[idx_111, 0, 0], 0)

        # By calculating distance and normalize by sqrt(2).
        np.testing.assert_almost_equal(matrix[idx_111, 0, 1:],
                                       np.array([1, 1, 1, 1, 2]) * np.pi / 2,
                                       decimal=4,)

if __name__ == '__main__':
    unittest.main()
