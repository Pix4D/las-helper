# -*- coding: utf-8 -*-

import pathlib
import helper_las
import helper_xyz

data_path_xyz = [
    "./pole_to_pole_m.xyz",
    "./pole_to_pole_feet.xyz",
    "./point_series.xyz",
    "./point_series_precision.xyz",
]


def analyse_xyz_files():
    for file in data_path_xyz:
        print(f"#################\n{file}\n")
        f = helper_xyz.XyzReader(file)
        f.print_points()
        print()


data_path_las = [
    "./pole_to_pole_m_CC.las",
    "./pole_to_pole_feet_CC.las",
    "./point_series_CC.las",
    "./point_series_precision_CC.las",
]


def analyse_las_files():
    for file in data_path_las:
        print(f"#################\n{file}\n")
        f = helper_las.LasHelper(file, "r")
        f.print_header_info()
        f.print_points()
        print()


data_path_mapper_las = [
    r"E:\datasets_hdd\valid_datasets\Mapper_Projects\Small_max_5GB\soda_power_lines\power_line\2_densification\point_cloud\power_line_group1_densified_point_cloud.las",
    r"E:\datasets_hdd\valid_datasets\Mapper_Projects\Medium_max_25GB\Electrica1\Linea Electrica\2_densification\point_cloud\Linea Electrica_group1_densified_point_cloud_part_1.las",
    r"E:\datasets_hdd\valid_datasets\Mapper_Projects\Medium_max_25GB\Electrica1\Linea Electrica\2_densification\point_cloud\Linea Electrica_group1_densified_point_cloud_part_2.las",
    r"E:\datasets_hdd\valid_datasets\Mapper_Projects\Large\Highway\Highway\2_densification\point_cloud\Highway_group1_densified_point_cloud.las",
]


def analyse_mapper_headers():
    for file in data_path_mapper_las:
        print(f"#################\n{pathlib.Path(file).name}\n")
        f = helper_las.LasHelper(file, "r")
        f.print_header_info()
        print()


analyse_las_files()
