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


class FWHM:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f'FWHM(x={round(self.x, 2)}, y={round(self.y, 2)}, xy={round(self.xy, 2)})'

    @property
    def xy(self) -> float:
        return (self.x + self.y)/2


class Gaussian2D:
    def __init__(self, x: int, y: int, w: int = 64, h: int = None):  # x = 3013, y = 1961
        self.x, self.y, self.w = x, y, w
        self.h = w if h is None else h
        self.c_x, self.c_y, self.sigma_x, self.sigma_y, self.a, self.b = (None for _ in range(6))

    @staticmethod
    def add_subplot(fig, X, Y, Z, angle: Tuple[float, float], position: Tuple[int, int, int], stride: int):
        ax = fig.add_subplot(*position, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, antialiased=True, cmap=cm.magma)

        ax.view_init(*angle)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.magma)

    @staticmethod
    def gaussian(xy: tuple, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """Function to fit, returns 2D gaussian function as 1D array"""
        x, y = xy
        x0 = float(x0)
        y0 = float(x0)
        g = offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))
        return g.ravel()

    @property
    def _fwhm(self) -> FWHM:
        return FWHM(np.abs(4*self.sigma_x*np.sqrt(-0.5*np.log(0.5))), np.abs(4*self.sigma_y*np.sqrt(-0.5*np.log(0.5))))

    def fit(self, path: str, imshow: bool = False) -> FWHM:  # https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7
        # read
        img: np.ndarray = cv2.imread(path, -1)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # crop
        img = img[self.y - self.h//2:self.y + self.h//2, self.x - self.w//2:self.x + self.w//2]
        if imshow:
            cv2.imshow(path, img)
            cv2.waitKey(0)

        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        x, y = np.meshgrid(x, y)
        # Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
        initial_guess = (img.shape[1]/2, img.shape[0]/2, 10, 10, 1, 0)
        # subtract background and rescale image into [0,1], with floor clipping
        bg = np.percentile(img, 5)
        img_scaled = np.clip((img - bg) / (img.max() - bg), 0, 1)
        p_opt, _ = opt.curve_fit(self.gaussian, (x, y), img_scaled.ravel(), p0=initial_guess, bounds=((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1), (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
        self.c_x, self.c_y, self.sigma_x, self.sigma_y, self.a, self.b = p_opt
        return self._fwhm

    def show(self, size: int = 256, figsize: Tuple[int, int] = (9, 3), stride: int = 3):
        x = np.linspace(-self.w/2, self.w/2, size)
        y = np.linspace(-self.h/2, self.h/2, size)
        x, y = np.meshgrid(x, y)
        z = (1 / (2 * np.pi * self.sigma_x * self.sigma_y) * np.exp(-(x ** 2 / (2 * self.sigma_x ** 2) + y ** 2 / (2 * self.sigma_y ** 2))))

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)
        self.add_subplot(fig, x, y, z, (40, 25), (1, 3, 1), stride)
        self.add_subplot(fig, x, y, z, (0, 0), (1, 3, 2), stride)
        self.add_subplot(fig, x, y, z, (90, 0), (1, 3, 3), stride)
        plt.show()


class Fit(Gaussian2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datetime: datetime
        self.t, self.fwhm = None, None

    @staticmethod
    def progressbar(progress: float, length: int = 30) -> str:
        parts = '_░▒▓█'
        done = int(progress * length)
        return (done * '█' + parts[round(4 * (length * progress - done))] + int((1 - progress) * length) * '_')[:length]

    def folder(self, folder: str, reference_folder: str, extension: str = '.tif', raw_extension: str = '.CR2', strpfmt: str = '%Y-%m-%d', info: bool = True, imshow: bool = False):
        self.t, self.fwhm = [], []
        folder_len = len(os.listdir(folder))
        self.datetime = datetime.strptime(folder.split('\\')[-1], strpfmt)

        if info:
            print(f"[Fit] fitting '{folder}'")
        for i, file in enumerate(os.listdir(folder)):
            if not file.endswith('.reg' + extension):
                continue
            if info:
                print(f'\r[Fit] fitting {file} | {self.progressbar(i / folder_len)} | {round(100 * i / folder_len, 2)}%\t', end='')
            '''https://pypi.org/project/ExifRead/'''
            with open(f'{reference_folder}\\{file.replace(".reg" + extension, raw_extension)}', 'rb') as f:
                tags = process_file(f)
            t = datetime.strptime(str(tags['EXIF DateTimeOriginal']), '%Y:%m:%d %H:%M:%S')
            fwhm = self.fit(folder + '\\' + file, imshow)
            self.t.append(t)
            self.fwhm.append(fwhm.xy)
        if info:
            print(f'\r[Fit] fitting | {self.progressbar(1)} | Done\t', end='')

    def show(self):
        plt.clf()
        plt.plot(self.t, self.fwhm, 'x', color='black', markersize=3)
        plt.show()

    def save(self, to_folder: str = r'D:\Ha_Jong\AAA_Pictures\2020\ASTROPHOTOGRAPHY\X_Cyg\[data]'):
        file = to_folder + '\\' + self.datetime.strftime('%Y-%m-%d') + '.fwhm'
        with open(file, 'w') as f:
            for t, fwhm in zip(self.t, self.fwhm):
                f.write(f'{str(t)},{fwhm}\n')


class Data:
    def __init__(self, folder: str):
        self.folder = folder
        self.t, self.fwhm = None, None

    def get(self, strpfmt: str = '%Y-%m-%d %H:%M:%S'):
        self.t, self.fwhm, now = [], [], datetime.now()

        for fwhm_file in os.listdir(self.folder):
            fwhm_day = []
            if not fwhm_file.endswith('.fwhm'):
                continue

            with open(self.folder + '\\' + fwhm_file) as f:
                lines = f.read().splitlines()
            for i, line in enumerate(lines):
                t, fwhm = line.split(',')
                if i == len(lines) // 2:
                    self.t.append((datetime.strptime(t, strpfmt) - now).days)
                fwhm_day.append(float(fwhm))
            self.fwhm.append(fwhm_day)

    def show(self):
        plt.clf()
        plt.xlabel('days')
        plt.ylabel('FWHM [px]')
        plt.boxplot(self.fwhm, positions=self.t)
        plt.show()


if __name__ == '__main__':
    data = Data(r'D:\Ha_Jong\AAA_Pictures\2020\ASTROPHOTOGRAPHY\X_Cyg\[data]')
    data.get()
    data.show()

    # fit = Fit(3013, 1961)
    # fit.folder(r'D:\Ha_Jong\AAA_Pictures\2020\ASTROPHOTOGRAPHY\X_Cyg\[reg]\2021-09-26', r'D:\Ha_Jong\AAA_Pictures\2020\ASTROPHOTOGRAPHY\X_Cyg\2021-09-26 - X Cyg')
    # fit.save()
    # fit.show()
