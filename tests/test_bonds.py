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


if __name__ == '__main__':
    unittest.main()
