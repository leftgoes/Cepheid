import os
from ast import literal_eval
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from lmfit import Model
from inspect import getsource, signature
from scipy.signal import find_peaks
import numpy as np
from typing import Callable


class polynomials:
    @classmethod
    def get(cls):
        return cls.linear, cls.quadratic, cls.cubic, cls.quartic, cls.quintic

    @staticmethod
    def linear(x, a, b):
        return a + b * x

    @staticmethod
    def quadratic(x, a, b, c):
        return a + b * x + c * x**2

    @staticmethod
    def cubic(x, a, b, c, d):
        return a + b * x + c * x**2 + d * x**3

    @staticmethod
    def quartic(x, a, b, c, d, f):
        return a + b * x + c * x**2 + d * x**3 + f * x**4

    @staticmethod
    def quintic(x, a, b, c, d, f, g):
        return a + b * x + c * x**2 + d * x**3 + f * x**4 + g * x**5


class Info:
    def __init__(self, name: str, folder: str = None):
        self.name = name
        self.folder = folder

    def __repr__(self):
        return f'Info({self.name})'

    def __getattr__(self, item):
        with open(self.folder + '\\' + self.name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith(item):
                return literal_eval('(' + line[line.find('=') + 2:] + ')')
        return AttributeError

    @property
    def number(self):
        return int(self.name[4:self.name.find('.Info.txt')])


class LMFit:
    def __init__(self, f: Callable):
        self.f = f
        self.data = ((1, 1) for _ in self.parameters)
        self.items = dict(zip(self.parameters, [var[1] for var in self.data]))

    def __dict__(self):
        return self.items

    def get_param(self, result):
        report = result.fit_report()
        lines = report[report.find('[[Variables]]') + 14:report.find('[[Correlations]]')].splitlines()
        parameters_string = '{'
        max_var_len = max(len(line[4:4 + line[4:].find(':')]) for line in lines)
        for line in lines:
            line = line[4:18 + max_var_len]
            parameters_string += f"'{line[:line.find(':')]}'{line[line.find(':'):]}, "
        self.items = literal_eval(parameters_string[:-2] + '}')

    def fit(self, x: list, y: list):
        model = Model(self.f)
        self.data = self.get_param(model.fit(y, x=x, **self.__dict__()))

    def q(self, x):
        return self.f(x, **self.__dict__())

    @property
    def parameters(self) -> list:
        return list(signature(self.f).parameters.keys())[1:]

    @property
    def returning(self):
        source = getsource(self.f)
        return source[source.find('return') + 7:-1]


class Stack:
    def __init__(self, folder: str, files: int = 90, target: str = None):
        self.folder = folder
        self.files = files
        self.target = target
        self.lmfit = LMFit(polynomials.linear)
        self.styles = {'color': ['orange', 'green', 'black'], 'dot': ['o', 'o', 'x'], 'size': [2, 2, 5]}
        self.args, self.maximum, self.x, self.x0, self.y, self.dark, self.flat, self.offset = (None for _ in range(8))

    @staticmethod
    def int2str(num: int, m: int = 4):
        string = str(num)
        return (m - len(string)) * '0' + string

    @staticmethod
    def round(x: float, significant: int = 0) -> int or float:
        power = int(np.log10(abs(x)))
        rounded = round(x, significant - power - 1)
        return rounded if power + 1 < significant else int(rounded)

    @staticmethod
    def string2raised(string: str) -> str:
        string = str(int(string))
        numbers, raised = '⁰¹²³⁴⁵⁶⁷⁸⁹', ''
        for char in string:
            if char == '-':
                raised += '⁻'
            else:
                raised += numbers[int(char)]
        return raised

    @property
    def fitted(self, *args, num: int = None, **kwargs) -> tuple[np.ndarray]:
        if num is None:
            num = len(self.x)
        self.lmfit.fit(self.x, self.y)
        x_poly = np.linspace(self.x[0], self.x[-1], num, *args, **kwargs)
        return x_poly, self.lmfit.q(x_poly)

    @property
    def function_string(self):
        raised = '⁰¹²³⁴⁵⁶⁷⁸⁹'
        f = self.lmfit.returning
        for key, value in self.lmfit.__dict__().items():
            f = f.replace(key, str(self.round(value, 4)))

        if self.lmfit.f is polynomials.linear:
            return f.replace(' * x', 'x')

        clean = ''
        while len(f) > 1 and f.find('**') != -1:
            index = f.find('**')
            clean += f[:index] + raised[int(f[index + 2])]
            f = f[index + 3:]
        for from_to in [('+ -', '- '), ('- -', '+ '), ('*', ''), ('  x', 'x')]:
            clean = clean.replace(*from_to)
        f = clean

        clean, index = '', f.find('e')
        while index != -1 and len(f) > 1:
            clean += f[:index] + '·10' + self.string2raised(f[index + 1:f[index:].find('x') + index])
            f0 = f[len(clean) - 1:]
            f = f0 if len(f0) > 2 else f[-2:]
            index = f.find('e')
        clean += f
        return clean

    @property
    def patches(self):
        return [mpatches.Patch(color='white', label=f'⌀stack = {self.round(self.sum, 5)}')] + [mpatches.Patch(color=self.styles['color'][2 - i], label=label) for i, label in enumerate(['stacking', f'{self.files} < Δmax', 'q(x) < ' + self.function_string])]

    @property
    def sum(self):
        stacking, score = self.use[2], 0
        for x, y in stacking:
            score += y
        return score/len(stacking)

    @property
    def use(self) -> tuple[set, set, list]:
        zero = {(x, y) for x, y in zip(self.x, self.y)}
        first = {(x, y) for x, y in zero if self.lmfit.q(x) < y}
        second = set()
        x1, x2 = self.use_range
        for x, y in first:
            if x1 <= x <= x2:
                second.add((x, y))
        second.intersection(first)
        return zero - first - second, first - second, sorted(second, key=lambda i: i[1], reverse=True)

    @property
    def use_range(self):
        m = self.maximum
        return m - self.files, m + self.files

    def get_data(self):
        infos = [Info(file, self.folder) for file in os.listdir(self.folder) if file.endswith('.Info.txt')]
        self.x0 = infos[0].number
        self.x, self.y = ([info.number - self.x0 for info in infos], [info.OverallQuality for info in infos])

    def get_maximum(self, fit: bool = True):
        if fit and self.lmfit.data is None:
            self.lmfit.fit(self.x, self.y)
        x, y = self.fitted

        peak_index = find_peaks(y)[0]
        if len(peak_index) == 0:
            peak_x, peak_y = None, 0
        else:
            peak_x = self.x[peak_index[0]]
            peak_y = self.lmfit.q(peak_x)

        if max(y) > peak_y:
            if y[0] > y[-1]:
                self.maximum = x[0] + self.files
            else:
                self.maximum = x[-1] - self.files
        elif peak_x < self.files:
            self.maximum = x[0] + self.files
        elif peak_x > x[-1] - self.files:
            self.maximum = x[-1] - self.files
        else:
            self.maximum = peak_x

    def show(self, fit: bool = True, annotate: int or set or list = None):
        if fit and self.lmfit.data is None:
            self.lmfit.fit(self.x, self.y)
        if annotate is None:
            annotate = {0}
        elif type(annotate) is int:
            annotate = {annotate}

        plt.clf()
        plt.title(self.folder.split('\\')[-1])
        plt.xlabel('Image')
        plt.ylabel('Quality q')
        plt.plot()

        x1, x2 = self.use_range
        plt.vlines(x1 + self.x0, self.lmfit.q(x1), max(self.y), linestyles='dotted')
        plt.vlines(x2 + self.x0, self.lmfit.q(x2), max(self.y), linestyles='dotted')

        for i, data in enumerate(self.use):
            for j, (x, y) in enumerate(data):
                if i == 2 and j in annotate:
                    plt.annotate(f'(IMG_{x + self.x0}, {y})', (x, y))
                else:
                    plt.plot(x + self.x0, y, self.styles['dot'][i], color=self.styles['color'][i], markersize=self.styles['size'][i])

        fit_x, fit_y = self.fitted
        plt.plot(fit_x + self.x0, fit_y, '--', color='red', linewidth=1)
        plt.legend(handles=self.patches, loc='lower left')
        plt.show()

    def create_file(self, to_file: str = None, sub_length: int = 10, **kwargs) -> str:
        filenames = [f'IMG_{self.int2str(x + self.x0)}.CR2' for x, y in sorted(self.use[2], key=lambda i: i[0])]
        string = 'DSS file list\nCHECKED\tTYPE\tFILE\n'
        for file in filenames:
            string += f'1\tlight\t{self.folder}\\{file}\n'

        if self.offset is not None:
            string += '1\toffset\t' + self.offset + '\n'
        if self.flat is not None:
            string += '1\tflat\t' + self.flat + '\n'
        if self.dark is not None:
            string += '1\tdark\t' + self.dark + '\n'

        other = '#WS#Software\\DeepSkyStacker\\FitsDDP|BayerPattern=4\n#WS#Software\\DeepSkyStacker\\FitsDDP|BlueScale=1.0000\n#WS#Software\\DeepSkyStacker\\FitsDDP|Brighness=1.0000\n#WS#Software\\DeepSkyStacker\\FitsDDP|DSLR=\n#WS#Software\\DeepSkyStacker\\FitsDDP|FITSisRAW=0\n#WS#Software\\DeepSkyStacker\\FitsDDP|ForceUnsigned=0\n#WS#Software\\DeepSkyStacker\\FitsDDP|Interpolation=Bilinear\n#WS#Software\\DeepSkyStacker\\FitsDDP|RedScale=1.0000\n#WS#Software\\DeepSkyStacker\\RawDDP|AHD=0\n#WS#Software\\DeepSkyStacker\\RawDDP|BlackPointTo0=0\n#WS#Software\\DeepSkyStacker\\RawDDP|BlueScale=1.0000\n#WS#Software\\DeepSkyStacker\\RawDDP|Brighness=1.0000\n#WS#Software\\DeepSkyStacker\\RawDDP|CameraWB=0\n#WS#Software\\DeepSkyStacker\\RawDDP|Interpolation=Bilinear\n#WS#Software\\DeepSkyStacker\\RawDDP|NoWB=0\n#WS#Software\\DeepSkyStacker\\RawDDP|RawBayer=0\n#WS#Software\\DeepSkyStacker\\RawDDP|RedScale=1.0000\n#WS#Software\\DeepSkyStacker\\RawDDP|SuperPixels=0\n#WS#Software\\DeepSkyStacker\\Register|ApplyMedianFilter=1\n#WS#Software\\DeepSkyStacker\\Register|DetectHotPixels=1\n#WS#Software\\DeepSkyStacker\\Register|DetectionThreshold=30\n#WS#Software\\DeepSkyStacker\\Register|PercentStack=80\n#WS#Software\\DeepSkyStacker\\Register|StackAfter=0\n#WS#Software\\DeepSkyStacker\\Stacking|AlignChannels=0\n#WS#Software\\DeepSkyStacker\\Stacking|AlignmentTransformation=0\n#WS#Software\\DeepSkyStacker\\Stacking|ApplyFilterToCometImages=1\n#WS#Software\\DeepSkyStacker\\Stacking|BackgroundCalibration=0\n#WS#Software\\DeepSkyStacker\\Stacking|BackgroundCalibrationInterpolation=1\n#WS#Software\\DeepSkyStacker\\Stacking|BadLinesDetection=0\n#WS#Software\\DeepSkyStacker\\Stacking|CometStackingMode=0\n#WS#Software\\DeepSkyStacker\\Stacking|CreateIntermediates=0\n#WS#Software\\DeepSkyStacker\\Stacking|DarkFactor=1.0000\n#WS#Software\\DeepSkyStacker\\Stacking|DarkOptimization=0\n#WS#Software\\DeepSkyStacker\\Stacking|Dark_Iteration=5\n#WS#Software\\DeepSkyStacker\\Stacking|Dark_Kappa=2.0000\n#WS#Software\\DeepSkyStacker\\Stacking|Dark_Method=2\n#WS#Software\\DeepSkyStacker\\Stacking|Debloom=0\n#WS#Software\\DeepSkyStacker\\Stacking|Flat_Iteration=5\n#WS#Software\\DeepSkyStacker\\Stacking|Flat_Kappa=2.0000\n#WS#Software\\DeepSkyStacker\\Stacking|Flat_Method=2\n#WS#Software\\DeepSkyStacker\\Stacking|HotPixelsDetection=1\n#WS#Software\\DeepSkyStacker\\Stacking|IntermediateFileFormat=1\n#WS#Software\\DeepSkyStacker\\Stacking|Light_Iteration=5\n#WS#Software\\DeepSkyStacker\\Stacking|Light_Kappa=2.0000\n#WS#Software\\DeepSkyStacker\\Stacking|Light_Method=4\n#WS#Software\\DeepSkyStacker\\Stacking|LockCorners=1\n#WS#Software\\DeepSkyStacker\\Stacking|Mosaic=0\n#WS#Software\\DeepSkyStacker\\Stacking|Offset_Iteration=5\n#WS#Software\\DeepSkyStacker\\Stacking|Offset_Kappa=2.0000\n#WS#Software\\DeepSkyStacker\\Stacking|Offset_Method=2\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_ColdDetection=500\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_ColdFilter=1\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_DetectCleanCold=0\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_DetectCleanHot=0\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_HotDetection=500\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_HotFilter=1\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_ReplaceMethod=1\n#WS#Software\\DeepSkyStacker\\Stacking|PCS_SaveDeltaImage=0\n#WS#Software\\DeepSkyStacker\\Stacking|PerChannelBackgroundCalibration=1\n#WS#Software\\DeepSkyStacker\\Stacking|PixelSizeMultiplier=1\n#WS#Software\\DeepSkyStacker\\Stacking|RGBBackgroundCalibrationMethod=2\n#WS#Software\\DeepSkyStacker\\Stacking|SaveCalibrated=0\n#WS#Software\\DeepSkyStacker\\Stacking|SaveCalibratedDebayered=0\n#WS#Software\\DeepSkyStacker\\Stacking|SaveCometImages=0\n#WS#Software\\DeepSkyStacker\\Stacking|UseDarkFactor=0'
        for key, value in kwargs.items():
            index = other.find(key.replace('_', '|'))
            other = other[:index] + str(value) + other[other[index:].find('\n'):]
        string += other

        if to_file is None:
            to_file = self.folder.split('\\')[-1]
        to_file += f'; {len(self.x)}%{len(filenames)}x{sub_length}s'
        path = ('C:\\Python\\Cepheids\\DSS\\' + to_file + '.txt').replace(' - ' + self.target, '').replace(' ', '_')
        with open(path, 'w') as f:
            f.write(string)
        return path

    def run(self, outputfolder_name: str = 'ASTROPHOTOGRAPHY', show: bool = True, create_file: bool = True, stack_files: bool = True, dss_folder: str = r'C:\Program Files\DeepSkyStacker (64 bit)', **kwargs):
        self.get_data()
        qs = []
        polynoms = polynomials.get()
        for polynomial in polynoms:
            self.lmfit = LMFit(polynomial)
            self.get_maximum()
            qs.append((self.sum, len(self.use[2])))
        max_index = qs.index(max(qs, key=lambda t: t[0]))
        self.lmfit = LMFit(polynoms[max_index])
        self.get_maximum()

        print('⌀stack:')
        for i, (q, n) in enumerate(qs):
            print(f' {i + 1}:\t{round(q, 2)}\t{n=}')
        print('q(x) = ' + self.function_string)

        if show:
            self.show()
        if create_file:
            paa = self.create_file(**kwargs)
            if stack_files:
                os.system(r'cd ' + dss_folder + ' & ' + r'DeepSkyStackerCL.exe /S ' + paa)


if __name__ == '__main__':
    stack = Stack(r'D:\Ha Jong\AAA_Pictures\2020\ASTROPHOTOGRAPHY\X Cyg\2021-09-26 - X Cyg', target='X Cyg')#
    stack.dark = r'D:\Ha Jong\AAA_Pictures\2020\Correction Frames\Darks\2021-11-05 - Darks 10s\MasterDark_ISO3200_10s.tif'
    stack.flat = r'D:\Ha Jong\AAA_Pictures\2020\Correction Frames\Flats\2021-09-26 - Flats\MasterFlat_ISO3200.tif'
    stack.offset = r'D:\Ha Jong\AAA_Pictures\2020\Correction Frames\Bias\2021-10-11 - Bias\MasterOffset_ISO3200.tif'
    stack.run(stack_files=False)

