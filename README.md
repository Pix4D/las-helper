# las-helper

Creation date: June, 17th 2019

Author: Jean-Philippe Loublanchès <jean-philippe.loublanches@pix4d.com>


## Description

Python modules for LAS and XYZ files manipulation.

* helper_las.py:            Read and write LAS files
* helper_xyz.py:            Read XYZ files
* convert_xyz_to_las.py:    Convert XYZ files to LAS files
* data_generator.py:        Generate point cloud and write in a LAS file

## Usage

**Opening and getting data in a LAS**

```
>>> import helper_las as hl

>>> f = hl.LasHelper("line_500km_sweden.las","r")

>>> f.print_header_info()
Header info "line_500km_sweden.las"
LAS format              : 1.2
Data format ID          : 3
Point count             : 248889312
Min values     [x, y, z]: [895543.034, 7498943.492, 0.0]
Max values     [x, y, z]: [1395543.034, 7499043.492, 0.0]
Offset factors [x, y, z]: [1145543.0, 7498993.0, 0.0]
Scale factors  [x, y, z]: [0.001, 0.001, 0.001]

>>> f.print_points()
Scaled points (float64):
[[ 895543.034        7498943.492              0.          ]
 [ 895543.4839999999 7498943.492              0.          ]
 [ 895543.934        7498943.492              0.          ]
 ...
 [1395542.134        7499043.492              0.          ]
 [1395542.584        7499043.492              0.          ]
 [1395543.034        7499043.492              0.          ]]
 
>>> a = f.get_points()

>>> a
array([[ 895543.034       , 7498943.492       ,       0.          ],
       [ 895543.4839999999, 7498943.492       ,       0.          ],
       [ 895543.934       , 7498943.492       ,       0.          ],
       ...,
       [1395542.134       , 7499043.492       ,       0.          ],
       [1395542.584       , 7499043.492       ,       0.          ],
       [1395543.034       , 7499043.492       ,       0.          ]])
```

**Creating a LAS file**
```
>>> import helper_las as hl
>>> import numpy as np
>>> data = np.reshape([[1.123, 2.123, 3.123],[4,5,6]], [2,3])

>>> data
array([[1.123, 2.123, 3.123],
       [4.   , 5.   , 6.   ]])

>>> f = hl.LasHelper("tmp.las","w")

>>> f.set_scaled_points(data)

>>> f.print_header_info()
Header info "tmp.las"
LAS format              : 1.2
Data format ID          : 3
Point count             : 2
Min values     [x, y, z]: [1.123, 2.123, 3.123]
Max values     [x, y, z]: [4.0, 5.0, 6.0]
Offset factors [x, y, z]: [3.0, 4.0, 5.0]
Scale factors  [x, y, z]: [0.001, 0.001, 0.001]

>>> f.set_color_all_points([200,0,0])

>>> f.close()
```

**Converting a XYZ file into a LAS file**

Cf. convert_xyz_to_las.py for an example

## Development

**Formatting of the Python code**

```
black --line-length=99 .
```