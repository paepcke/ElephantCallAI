import unittest
import numpy as np

from gun_data.VisualizationUtils import create_visualization_image


class VisualizationUtilsTest(unittest.TestCase):
    def test_visualize(self):
        # This test is not automated, you must inspect the output visually
        xdim = 525
        ydim = 95

        x = np.arange(xdim).reshape((xdim, 1)).astype(np.float32)
        y = np.arange(ydim).reshape((ydim, 1)).astype(np.float32)
        data = np.matmul(y, x.T)
        data = np.sqrt(data)
        create_visualization_image(data, [data * 0.8, data * 0.6], "Test Image", None, "blah")


if __name__ == '__main__':
    unittest.main()
