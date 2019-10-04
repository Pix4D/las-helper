# -*- coding: utf-8 -*-

import numpy as np
import helper_las as hl

start_zone_utm_32n = [-1206117.77, 4021309.83]
end_zone_utm_32n = [1295389.07, 8051813.30]

start_position = [895543.033814, 7498943.491609]
end_position = [1201102.728944, 7981165.873143]


def write_data(file_path, data):
    f = hl.LasHelper(file_path, "w")
    f.set_scaled_points(data)
    f.print_header_info()
    f.set_color_all_points([200, 0, 0])
    f.close()


def create_point_cloud_square_width(start, width, point_per_axis):
    x = np.linspace(start[0], start[0] + width, point_per_axis)
    y = np.linspace(start[1], start[1] + width, point_per_axis)
    xx, yy, = np.meshgrid(x, y)
    return np.vstack((xx.flatten(), yy.flatten(), np.zeros(len(xx.flatten())))).transpose()


def create_point_cloud_empty_square_width(start, width, point_per_axis):
    x = np.linspace(start[0], start[0] + width, point_per_axis)
    y = np.linspace(start[1], start[1] + width, point_per_axis)
    xx = np.hstack((x, [x[0]] * point_per_axis, x, [x[-1]] * point_per_axis))
    yy = np.hstack(([y[0]] * point_per_axis, y, [y[-1]] * point_per_axis, y))
    return np.vstack((xx.flatten(), yy.flatten(), np.zeros(len(xx.flatten())))).transpose()


def create_point_cloud_square(start, end, point_per_axis):
    x = np.linspace(start[0], end[0], point_per_axis)
    y = np.linspace(start[1], end[1], point_per_axis)
    xx, yy, = np.meshgrid(x, y)
    return np.vstack((xx.flatten(), yy.flatten(), np.zeros(len(xx.flatten())))).transpose()


def create_point_cloud_line(start, line_length_m, line_width_m):
    # For the purpose of testing, we want to have an overlap between points
    # So that the point cloud looks like a surface
    # Survey's point size is at most 50cm currently
    # Therefore, spacing of points should be less than 50cm
    spacing = 0.45
    point_per_x = int(line_length_m / spacing) + 2
    point_per_y = int(line_width_m / spacing) + 2

    x, step = np.linspace(start[0], start[0] + line_length_m, point_per_x, endpoint=False, retstep=True)
    if step > spacing:
        raise RuntimeError(f"Computed step ({step}m) is bigger than spacing ({spacing}m)")

    y, step = np.linspace(start[1], start[1] + line_width_m, point_per_y, endpoint=False, retstep=True)
    if step > spacing:
        raise RuntimeError(f"Computed step ({step}m) is bigger than spacing ({spacing}m)")

    xx, yy, = np.meshgrid(x, y)
    return np.vstack((xx.flatten(), yy.flatten(), np.zeros(len(xx.flatten())))).transpose()


# write_data("square_sweden.las", create_point_cloud_square(start_position, end_position, 100))
# write_data("square_4km_sweden.las", create_point_cloud_square_width(start_position, 4000, 10000))
# write_data("square_40m_sweden.las", create_point_cloud_square_width(start_position, 40, 100))
# write_data("empty_square_4km_sweden.las", create_point_cloud_empty_square_width(start_position, 4000, 10000))
# write_data("line_5km_sweden.las", create_point_cloud_line(start_position, 5000, 500))
# write_data("line_50km_sweden.las", create_point_cloud_line(start_position, 50000, 500))
write_data("line_500km_sweden.las", create_point_cloud_line(start_position, 500000, 100))
# write_data("line_100km_sweden.las", create_point_cloud_line(start_position, 100000, 60))
