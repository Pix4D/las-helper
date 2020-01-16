json_pipeline = """
[
    {
        "type":"readers.las",
        "filename":"square_plane_5_50_100_500.las"
    },
    {
        "type":"writers.las",
        "filename":"square_plane_5_50_100_500.las",
        "minor_version": 2,
        "a_srs":"EPSG:32636"
    }
]
"""
