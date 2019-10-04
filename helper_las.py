# -*- coding: utf-8 -*-

import laspy
import numpy as np

np.set_printoptions(precision=9, suppress=True, floatmode="unique")


class LasHelper:
    def __init__(self, file_path, mode):
        self.__file_path = file_path
        if mode == "r":
            self.__las_file = laspy.file.File(file_path, mode="r")

        elif mode == "w":
            header = laspy.header.Header(file_version=1.2, point_format=3)
            self.__las_file = laspy.file.File(file_path, mode="w", header=header)
            self.__data_set_scaled = None
            self.__las_file.header.scale = [1e-3, 1e-3, 1e-3]
            self.__las_file.header.offset = [0.0, 0.0, 0.0]

        else:
            raise RuntimeError(f"Unsupported mode {mode}")

        self.__header = self.__las_file.header

    def print_header_info(self):
        print(f'Header info "{self.__file_path}"')
        print(f"LAS format              : {self.__header.version}")
        print(f"Data format ID          : {self.__header.data_format_id}")
        print(f"Point count             : {self.get_point_count()}")
        print(f"Min values     [x, y, z]: {self.__header.min}")
        print(f"Max values     [x, y, z]: {self.__header.max}")
        print(f"Offset factors [x, y, z]: {self.__header.offset}")
        print(f"Scale factors  [x, y, z]: {self.__header.scale}")

    def get_points(self):
        return np.vstack((self.__las_file.x, self.__las_file.y, self.__las_file.z)).transpose()

    def print_raw_points(self):
        print("Not scaled points (int32)")
        print(self.__las_file.points)

    def print_points(self):
        data_set_scaled = self.get_points()
        print(f"Scaled points ({data_set_scaled.dtype}):")
        print(f"{data_set_scaled}")

    def set_scaled_points(self, points_scaled):
        self.__las_file.header.min = [
            round(min(points_scaled[:, 0]), 3),
            round(min(points_scaled[:, 1]), 3),
            round(min(points_scaled[:, 2]), 3),
        ]

        self.__las_file.header.max = [
            round(max(points_scaled[:, 0]), 3),
            round(max(points_scaled[:, 1]), 3),
            round(max(points_scaled[:, 2]), 3),
        ]

        offset = self.__las_file.header.offset = [
            round((x + y) / 2)
            for x, y in zip(self.__las_file.header.min, self.__las_file.header.max)
        ]
        scale = self.__las_file.header.scale

        self.__las_file.X = [round(x) for x in ((points_scaled[:, 0] - offset[0]) / scale[0])]
        self.__las_file.Y = [round(x) for x in ((points_scaled[:, 1] - offset[1]) / scale[1])]
        self.__las_file.Z = [round(x) for x in ((points_scaled[:, 2] - offset[2]) / scale[2])]

    def get_point_count(self):
        return len(self.__las_file.points)

    def close(self):
        self.__las_file.close()

    def set_color_all_points(self, color_rgb):
        point_count = len(self.__las_file.get_red())
        self.__las_file.set_red([color_rgb[0]] * point_count)
        self.__las_file.set_green([color_rgb[1]] * point_count)
        self.__las_file.set_blue([color_rgb[2]] * point_count)


def test_las_export_and_import():
    out_file_path = "tmp.las"

    input_points = np.array(
        [
            [-3.123, -30.123, -300.123],
            [-2.123, -20.123, -200.123],
            [-1.123, -10.123, -100.123],
            [1.123, 10.123, 100.123],
            [2.123, 20.123, 200.123],
            [3.123, 30.123, 300.123],
        ],
        np.double,
    )

    print(f"Input points ({input_points.dtype}):")
    print(f"{input_points}")

    # Write to LAS file
    c1 = LasHelper(out_file_path, "w")
    c1.set_scaled_points(input_points)
    print()
    print(f"Input points, imported in LasHelper:")
    c1.print_header_info()
    c1.print_raw_points()
    c1.print_points()
    print()
    print(f"Export points in {out_file_path}")
    c1.close()

    # Read LAS
    print()
    print(f"Import points from {out_file_path}")
    c2 = LasHelper(out_file_path, "r")
    print()
    print(f"Points imported from LAS file):")
    c2.print_header_info()
    c2.print_raw_points()
    c2.print_points()

    # Compare
    print("\nComparing the 2 arrays")
    if np.allclose(input_points, c2.get_points()) is True:
        print("Test successful!")
        print("Data could be exported to a LAS file and imported again without data corruption")
    else:
        print("Test failed!")
        print("Array differences (output - input):")
        print(c2.get_points() - input_points)
