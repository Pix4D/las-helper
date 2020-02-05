import os
import shutil
import struct

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
        print(f"Offset: {offset}")
        scale = self.__las_file.header.scale
        print(f"Scale: {scale}")

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


def test_las_export_and_import(out_file_path):
    input_points = np.array(
        [
            [-300.123, -30.123, -3.123],
            [-200.123, -20.123, -2.123],
            [-100.123, -10.123, -1.123],
            [100.123, 10.123, 1.123],
            [200.123, 20.123, 2.123],
            [300.123, 30.123, 3.123],
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


def print_file_bytes(file_path, start_byte_pos, length):
    if length < 0:
        return

    if file_path == "" or os.path.isfile(file_path) is not True:
        return

    with open(file_path, "rb") as f:
        f.read(start_byte_pos)
        tmp = f.read(length)
        print(f"Bytes [{start_byte_pos}-{start_byte_pos+length-1}]:", tmp)


def read_bytes(file_path, start_byte_pos, length):
    if length < 0:
        return

    if file_path == "" or os.path.isfile(file_path) is not True:
        return

    with open(file_path, "rb") as f:
        f.read(start_byte_pos)
        return f.read(length)


def write_bytes(file_path, start_byte_pos, byte_str):
    if len(byte_str) <= 0:
        return

    if file_path == "" or os.path.isfile(file_path) is not True:
        return

    with open(file_path, "r+b") as f:
        f.seek(start_byte_pos)
        f.write(byte_str)


# LAS format 1.2
# Source: http://www.asprs.org/a/society/committees/standards/asprs_las_format_v12.pdf
#
# Item                                      Format              Size       Offset   Required
# File Signature (“LASF”)                   char[4]             4 bytes    0        *
#  File Source ID unsigned                  short               2 bytes    4        *
#  Global Encoding unsigned                 short               2 bytes    6        *
#  Project ID - GUID data 1                 unsigned long       4 bytes    8
#  Project ID - GUID data 2                 unsigned short      2 byte     12
#  Project ID - GUID data 3                 unsigned short      2 byte     14
#  Project ID - GUID data 4                 unsigned char[8]    8 bytes    16
# Version Major                             unsigned char       1 byte     24       *
# Version Minor                             unsigned char       1 byte     25       *
#  System Identifier                        char[32]            32 bytes   26       *
# Generating Software                       char[32]            32 bytes   58       *
#  File Creation Day of Year                unsigned short      2 bytes    90
#  File Creation Year                       unsigned short      2 bytes    92
# Header Size                               unsigned short      2 bytes    94       *
# Offset to point data                      unsigned long       4 bytes    96       *
# Number of Variable Length Records         unsigned long       4 bytes    100      *
# Point Data Format ID (0-99 for spec)      unsigned char       1 byte     104      *
# Point Data Record Length                  unsigned short      2 bytes    105      *
# Number of point records                   unsigned long       4 bytes    107      *
# Number of points by return                unsigned long[5]    20 bytes   111      *
# X scale factor                            double              8 bytes    131      *
# Y scale factor                            double              8 bytes    139      *
# Z scale factor                            double              8 bytes    147      *
# X offset                                  double              8 bytes    155      *
# Y offset                                  double              8 bytes    163      *
# Z offset                                  double              8 bytes    171      *
# Max X                                     double              8 bytes    179      *
# Min X                                     double              8 bytes    187      *
# Max Y                                     double              8 bytes    195      *
# Min Y                                     double              8 bytes    203      *
# Max Z                                     double              8 bytes    211      *
# Min Z                                     double              8 bytes    219      *


class HeaderPropertyDescriptor:
    # _format should follow this standard: https://docs.python.org/3/library/struct.html
    def __init__(self, _start_byte, _length, _format):
        self.start_byte = _start_byte
        self.length = _length
        self.format = _format


property_descriptors = {
    "file_signature": HeaderPropertyDescriptor(0, 4, "s"),
    "version_major": HeaderPropertyDescriptor(24, 1, "B"),
    "version_minor": HeaderPropertyDescriptor(25, 1, "B"),
    "offset_to_point_data": HeaderPropertyDescriptor(96, 4, "L"),
    "number_of_point_records": HeaderPropertyDescriptor(107, 4, "L"),
    "x_scale": HeaderPropertyDescriptor(131, 8, "d"),
    "y_scale": HeaderPropertyDescriptor(139, 8, "d"),
    "z_scale": HeaderPropertyDescriptor(147, 8, "d"),
    "x_offset": HeaderPropertyDescriptor(155, 8, "d"),
    "y_offset": HeaderPropertyDescriptor(163, 8, "d"),
    "z_offset": HeaderPropertyDescriptor(171, 8, "d"),
    "x_max": HeaderPropertyDescriptor(179, 8, "d"),
    "x_min": HeaderPropertyDescriptor(187, 8, "d"),
    "y_max": HeaderPropertyDescriptor(195, 8, "d"),
    "y_min": HeaderPropertyDescriptor(203, 8, "d"),
    "z_max": HeaderPropertyDescriptor(211, 8, "d"),
    "z_min": HeaderPropertyDescriptor(219, 8, "d"),
}


class Pix4DLasHeader:
    """A home made LAS header reader and writer"""

    def __init__(self, file_path=""):
        self.file_path = file_path
        if self.file_signature != "LASF":
            raise Exception("Invalid file signature. Cannot read file.")
        if self.major_version != 1:
            raise Exception("Invalid major version. Support only 1.")
        if self.minor_version != 2:
            raise Exception("Invalid minor version. Support only 2.")

    def read_property(self, property_descriptor):
        tmp = read_bytes(
            self.file_path, property_descriptor.start_byte, property_descriptor.length
        )
        if property_descriptor.format != "s":
            return struct.unpack(property_descriptor.format, tmp)[0]
        else:
            return tmp.decode()

    def write_property(self, property_descriptor, value):
        if isinstance(value, str) is True:
            byte_str = value.encode()
        else:
            byte_str = struct.pack(property_descriptor.format, value)
        write_bytes(self.file_path, property_descriptor.start_byte, byte_str)

    file_signature = property(
        lambda self: self.read_property(property_descriptors["file_signature"])
    )
    major_version = property(
        lambda self: self.read_property(property_descriptors["version_major"])
    )
    minor_version = property(
        lambda self: self.read_property(property_descriptors["version_minor"])
    )
    offset_to_point_data = property(
        lambda self: self.read_property(property_descriptors["offset_to_point_data"])
    )
    number_of_point_records = property(
        fget=lambda self: self.read_property(property_descriptors["number_of_point_records"]),
        fset=lambda self, value: self.write_property(
            property_descriptors["number_of_point_records"], value
        ),
    )
    x_scale = property(
        lambda self: self.read_property(property_descriptors["x_scale"]),
        lambda self, value: self.write_property(property_descriptors["x_scale"], value),
    )
    y_scale = property(
        lambda self: self.read_property(property_descriptors["y_scale"]),
        lambda self, value: self.write_property(property_descriptors["y_scale"], value),
    )
    z_scale = property(
        lambda self: self.read_property(property_descriptors["z_scale"]),
        lambda self, value: self.write_property(property_descriptors["z_scale"], value),
    )
    x_offset = property(
        lambda self: self.read_property(property_descriptors["x_offset"]),
        lambda self, value: self.write_property(property_descriptors["x_offset"], value),
    )
    y_offset = property(
        lambda self: self.read_property(property_descriptors["y_offset"]),
        lambda self, value: self.write_property(property_descriptors["y_offset"], value),
    )
    z_offset = property(
        lambda self: self.read_property(property_descriptors["z_offset"]),
        lambda self, value: self.write_property(property_descriptors["z_offset"], value),
    )
    x_max = property(
        lambda self: self.read_property(property_descriptors["x_max"]),
        lambda self, value: self.write_property(property_descriptors["x_max"], value),
    )
    x_min = property(
        lambda self: self.read_property(property_descriptors["x_min"]),
        lambda self, value: self.write_property(property_descriptors["x_min"], value),
    )
    y_max = property(
        lambda self: self.read_property(property_descriptors["y_max"]),
        lambda self, value: self.write_property(property_descriptors["y_max"], value),
    )
    y_min = property(
        lambda self: self.read_property(property_descriptors["y_min"]),
        lambda self, value: self.write_property(property_descriptors["y_min"], value),
    )
    z_max = property(
        lambda self: self.read_property(property_descriptors["z_max"]),
        lambda self, value: self.write_property(property_descriptors["z_max"], value),
    )
    z_min = property(
        lambda self: self.read_property(property_descriptors["z_min"]),
        lambda self, value: self.write_property(property_descriptors["z_min"], value),
    )


def test_Pix4DLasHelper(in_file_path):
    lr = Pix4DLasHeader(in_file_path)
    lh = laspy.file.File(in_file_path, mode="r")
    assert lr.offset_to_point_data == lh.header.data_offset
    assert lr.major_version == lh.header.major_version
    assert lr.minor_version == lh.header.minor_version
    assert lr.number_of_point_records == lh.header.records_count
    assert lr.x_scale == lh.header.scale[0]
    assert lr.y_scale == lh.header.scale[1]
    assert lr.z_scale == lh.header.scale[2]
    assert lr.x_offset == lh.header.offset[0]
    assert lr.y_offset == lh.header.offset[1]
    assert lr.z_offset == lh.header.offset[2]
    assert lr.x_max == lh.header.max[0]
    assert lr.x_min == lh.header.min[0]
    assert lr.y_max == lh.header.max[1]
    assert lr.y_min == lh.header.min[1]
    assert lr.z_max == lh.header.max[2]
    assert lr.z_min == lh.header.min[2]


def merge_las_files(las_file_path_1, las_file_path_2, output_path):
    lr1 = Pix4DLasHeader(las_file_path_1)
    lr2 = Pix4DLasHeader(las_file_path_2)

    print("Checking if files offsets and scales match")
    assert lr1.x_scale == lr2.x_scale
    assert lr1.y_scale == lr2.y_scale
    assert lr1.z_scale == lr2.z_scale
    assert lr1.x_offset == lr2.x_offset
    assert lr1.y_offset == lr2.y_offset
    assert lr1.z_offset == lr2.z_offset

    print("Copy file 1")
    shutil.copyfile(las_file_path_1, output_path)

    print("Add Point Data Records from file 2")
    # print(f"File 2 size: {os.path.getsize(las_file_path_2)}")
    # print(f"File 2 start of point data records: {lr2.read_offset_to_point_data()}")
    cc = 0
    with open(output_path, "ab") as f_out:
        with open(las_file_path_2, "rb") as f_in:
            # Moves fd to start of point data records
            f_in.seek(lr2.offset_to_point_data)
            tmp = f_in.read(1000)
            while tmp:
                n = f_out.write(tmp)
                cc += n
                tmp = f_in.read(1000)
    print(f"Written {cc} bytes in new file")

    # Set Point Record Count
    lh_out = Pix4DLasHeader(output_path)

    lh_out.number_of_point_records = lr1.number_of_point_records + lr2.number_of_point_records

    lh_out.x_max = max(lr1.x_max, lr2.x_max)
    lh_out.x_min = min(lr1.x_min, lr2.x_min)
    lh_out.y_max = max(lr1.y_max, lr2.y_max)
    lh_out.y_min = min(lr1.y_min, lr2.y_min)
    lh_out.z_max = max(lr1.z_max, lr2.z_max)
    lh_out.z_min = min(lr1.z_min, lr2.z_min)


def test_merge_las_files():
    pc_path_1 = "merge_small_1.las"
    pc_path_2 = "merge_small_2.las"
    out_path = "out.las"

    merge_las_files(pc_path_1, pc_path_2, out_path)

    lh1 = Pix4DLasHeader(pc_path_1)
    lh2 = Pix4DLasHeader(pc_path_2)
    lh_out = Pix4DLasHeader(out_path)
    assert (
        lh_out.number_of_point_records == lh1.number_of_point_records + lh2.number_of_point_records
    )
    assert lh_out.x_max == max(lh1.x_max, lh2.x_max)
    assert lh_out.x_min == min(lh1.x_min, lh2.x_min)
    assert lh_out.y_max == max(lh1.y_max, lh2.y_max)
    assert lh_out.y_min == min(lh1.y_min, lh2.y_min)
    assert lh_out.z_max == max(lh1.z_max, lh2.z_max)
    assert lh_out.z_min == min(lh1.z_min, lh2.z_min)


def append_las_files(las_file_path_1, las_file_path_2):
    lr1 = Pix4DLasHeader(las_file_path_1)
    lr2 = Pix4DLasHeader(las_file_path_2)
    initial_point_count = lr1.number_of_point_records

    print("Checking if files offsets and scales match")
    assert lr1.x_scale == lr2.x_scale
    assert lr1.y_scale == lr2.y_scale
    assert lr1.z_scale == lr2.z_scale
    assert lr1.x_offset == lr2.x_offset
    assert lr1.y_offset == lr2.y_offset
    assert lr1.z_offset == lr2.z_offset

    print("Add Point Data Records from file 2")
    # print(f"File 2 size: {os.path.getsize(las_file_path_2)}")
    # print(f"File 2 start of point data records: {lr2.read_offset_to_point_data()}")
    cc = 0
    with open(las_file_path_1, "ab") as f_out:
        with open(las_file_path_2, "rb") as f_in:
            # Moves fd to start of point data records
            f_in.seek(lr2.offset_to_point_data)
            tmp = f_in.read(1000)
            while tmp:
                n = f_out.write(tmp)
                cc += n
                tmp = f_in.read(1000)
    print(f"Written {cc} bytes in new file")

    # Set Point Record Count
    lr1.number_of_point_records += lr2.number_of_point_records

    lr1.x_max = max(lr1.x_max, lr2.x_max)
    lr1.x_min = min(lr1.x_min, lr2.x_min)
    lr1.y_max = max(lr1.y_max, lr2.y_max)
    lr1.y_min = min(lr1.y_min, lr2.y_min)
    lr1.z_max = max(lr1.z_max, lr2.z_max)
    lr1.z_min = min(lr1.z_min, lr2.z_min)


def test_append_las_file(pc_path_1):
    pc_path_2 = "merge_small_2.las"

    lh1 = LasHelper(pc_path_1, "r")
    initial_point_count = lh1.get_point_count()

    append_las_files(pc_path_1, pc_path_2)

    lh1 = Pix4DLasHeader(pc_path_1)
    lh2 = Pix4DLasHeader(pc_path_2)
    assert lh1.number_of_point_records() == initial_point_count + lh2.number_of_point_records()
    assert lh1.x_max == max(lh1.x_max, lh2.x_max)
    assert lh1.x_min == min(lh1.x_min, lh2.x_min)
    assert lh1.y_max == max(lh1.y_max, lh2.y_max)
    assert lh1.y_min == min(lh1.y_min, lh2.y_min)
    assert lh1.z_max == max(lh1.z_max, lh2.z_max)
    assert lh1.z_min == min(lh1.z_min, lh2.z_min)


if __name__ == "__main__":
    # Unit tests
    out_file_path = "tmp.las"
    # test_las_export_and_import(out_file_path)
    # test_Pix4DLasHelper(out_file_path)
    # test_merge_las_files()
    # test_append_las_file(out_file_path)

    file1_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\AerialWasatchLidar_WestSubset_2367MioPoints.las"
    file2_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge6.las"
    out_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge123456.las"
    merge_las_files(file1_path, file2_path, out_path)

    # file1_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge123.las"
    # file2_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge3.las"
    # append_las_files(file1_path, file2_path)
