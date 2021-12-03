import cv2
import numpy as np
import scipy.optimize as opt
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
from exifread import process_file
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter


def circle(radius: int):
    arr = np.zeros((2 * radius, 2 * radius))
    for i in range(2 * radius):
        for j in range(2 * radius):
            if np.sqrt((i - radius - 1) ** 2 + (-j + radius - 1) ** 2) < radius:
                arr[i, j] = 1
    return gaussian_filter(arr, sigma=1)


class Star:
    def __init__(self, x: int, y: int, aperture: int, magnitude: float = None, name: str = None, ):
        self.x = x
        self.y = y
        self.aperture = aperture #40 if aperture == 40 else 30
        self.circle = circle(self.aperture)
        self.magnitude = magnitude
        self.name = name


class Flux:
    def __init__(self, folder: str, target_star: Star, reference_stars: list[Star, ...]):
        self.folder = folder
        self.target = target_star
        self.reference = reference_stars
        self.img: np.ndarray

    def from_star(self, img: np.ndarray, star: Star):
        arr = img[star.y - star.aperture:star.y + star.aperture, star.x - star.aperture:star.x + star.aperture]
        k = np.kron(arr, np.ones((12, 12)))
        k /= k.max()
        k *= 255
        self.imshow(k.astype(np.uint8))
        return np.sum(arr * star.circle)

    def imshow(self, arr: np.ndarray = None):
        arr = self.img if arr is None else arr
        cv2.imshow('arr', arr)
        cv2.waitKey()

    def get(self, path: str, sub_length: int = 10, color: int = 1, percentile: float = 70.0):
        self.img: np.ndarray = cv2.imread(path, -1)[:, :, color]
        pixel_values = []

        background = round(np.percentile(self.img[np.where(self.img != 0)], percentile))
        img = self.img.clip(background)  # subtract background from image
        img -= img.min()

        j_target = self.from_star(img, self.target)
        for r_star in self.reference:
            j = self.from_star(img, r_star)
            mag = r_star.magnitude - 2.512 * np.log10(j_target/j)  # http://www.remote-astrophotography.com/Photometry/Photometry.html
            print(f'{r_star.magnitude = }, {r_star.aperture = }, {mag = }')


if __name__ == '__main__':
    references = [Star(2846, 1943, 12, 9.06, 'SAO 7042'),
                  Star(3019, 2146, 10, 10.88),
                  Star(3985, 2530, 16, 6.63, 'V568 Cyg'),
                  Star(3609, 2305, 10, 9.63, 'TYC 2695-1517-1'),
                  Star(2239, 1841, 10, 9.25, 'SAO 70440'),
                  Star(4604, 2100, 16, 7.47, 'HD 197310')]
    flux = Flux('', Star(3012, 1961, 20, name='X Cyg'), references)
    flux.get('IMG_1154.reg.tif')
