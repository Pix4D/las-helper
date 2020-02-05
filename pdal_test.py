import pdal
import pyproj
import pdal_pipeline

pipeline = pdal.Pipeline(pdal_pipeline.json_pipeline)
pipeline.validate()  # check if our JSON and options were good
pipeline.loglevel = 8  # really noisy
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log
