# -*- coding: utf-8 -*-

import pathlib
import helper_las
import helper_xyz


data_path_xyz = [  #'./pole_to_pole_m.xyz',
    #'./pole_to_pole_feet.xyz',
    #'./point_series.xyz',
    #'./point_series_precision.xyz'
    "./point_series_1.xyz"
]

for file in data_path_xyz:
    print(f"#################\n{file}\n")
    # Import XYZ file
    c = helper_xyz.XyzReader(file)
    print("\nPrint points from XYZ structure")
    c.print_points()

    # Write to LAS file
    out_file_path = pathlib.Path(file).stem + ".las"
    out = helper_las.LasHelper(out_file_path, "w")
    out.set_scaled_points(c.points())
    print("\nPrint points from LAS structure")
    out.print_points()
    out.close()

    f = helper_las.LasHelper(out_file_path, "r")
    print()
    f.print_header_info()
    print()
    f.print_points()
