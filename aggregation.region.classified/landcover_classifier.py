import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import math
import platform


rootdir = os.path.dirname(__file__)

landcover_directory = r"\\akif.internal\public\z_resources\im-wb\landcove_layers"

landcover_classes_csv = os.path.join(rootdir, 'im.nca.postprocessing\aggregation.region.classified\tmp')

def get_raster_data(path):
    """
    get_raster_data gets the addresses of all the raster files ("*.tif")
    stored in the directory specified by "path". Each raster file should
    correspond to a different year for the analysis.

    :param path: directory containing the raster files for the density observable
    to aggregate.
    :return: a list storing the addresses of all the raster files.
    """

    file_list = []
    for file in os.listdir(path):
        # Iterate over all the files in the specified directory.
        if ".tif" in file:
            # Process the file if it has a .tif format.
            if platform.system() is "Windows":
                address = os.path.join(path, file).replace("/","\\")
            else:
                address = os.path.join(path, file).replace("\\","/")
                #build the path according the OS running the script

            if address not in file_list:
                # Add the file address to the list if it had not been added before.
                file_list.append(address)
        else:
            pass

    return file_list.sort()

landcover_list = get_raster_data(landcover_directory)

