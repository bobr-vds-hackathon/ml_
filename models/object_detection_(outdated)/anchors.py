import itertools
from math import sqrt
import numpy as np


def _compute_sizes():
    areas = [128*128, 256*256, 512*512]
    x_aspects = [0.5, 1.0, 2.0]

    heights = np.array([x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])
    widths = np.array([sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])

    return np.vstack([heights, widths]).T


def generate_maps(img_size, feature_pixels):
    pass