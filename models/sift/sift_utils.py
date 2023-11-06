import cv2
import numpy
from sift import Sift, compute_sift


def image_resize(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def image_to_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


