# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(precision=9, suppress=True, floatmode="unique")


class XyzReader:
    def __init__(self, xyz_file_path):
        self.__data_set = np.genfromtxt(xyz_file_path, delimiter=",", usecols=(0, 1, 2))

    def print_points(self):
        print(f"Points ({self.__data_set.dtype}):")
        print(f"{self.__data_set}")

    def points(self):
        return self.__data_set
