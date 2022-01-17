import unittest
import matplotlib.pyplot as plt


from gun_data.models.positional.SinusoidalPositionalEncoding import SinusoidalPositionalEncoding


class SinusoidalPositionalEncodingTest(unittest.TestCase):
    def visualize_encodings(self):
        # This test requires manual inspection of the output to verify the plot looks correct.
        enc = SinusoidalPositionalEncoding(94, 100)
        plt.imshow(enc.positional_encodings.detach().numpy())
        plt.show()


if __name__ == '__main__':
    unittest.main()
