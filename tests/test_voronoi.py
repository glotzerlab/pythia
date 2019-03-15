import numpy as np
import freud
import pythia
import unittest


class TestVoronoi(unittest.TestCase):
    def test_angle_histogram(self):
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

        bins = 180
        descriptors = pythia.voronoi.angle_histogram(
            box, positions, bins, buffer_distance=None, area_weight_mode='product')

        idx_111 = 13

        # Test angles of voronoi polyhedra (still regular cube) to have only 0, 90, 180 degrees.
        # The 180 degrees occurs due to inability to discern the +-0 angle within float precision.
        np.testing.assert_equal(np.where(descriptors[idx_111])[0],
                                np.array([0, bins//2, bins-1]))

if __name__ == '__main__':
    unittest.main()
