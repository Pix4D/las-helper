import helper_las

# file1_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\AerialWasatchLidar_WestSubset_2367MioPoints.las"
# file2_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge6.las"
# out_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge123456.las"
# helper_las.merge_las_files(file1_path, file2_path, out_path)

file1_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge123456789.las"
file2_path = r"E:\datasets_hdd\valid_datasets\LAS_only\AerialWasatchLidarSubset\merge10.las"
helper_las.append_las_files(file1_path, file2_path, fill_up=True)
