import os
import geopandas as gpd
from osgeo import gdal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import platform
import itertools

import sys
sys.path.insert(1, './tmp')

import gdal_calc


"""Data inputs"""
# landcover_directory = r"\\akif.internal\public\z_resources\im-wb\landcove_layers" #from the VM
landcover_directory = "Z:\z_resources\im-wb\landcover_layers_copy" #from my computer
# landcover_directory = r"C:\Users\admin\Downloads\landcover_layers" #in local VM
landcover_classes_csv = "./tmp/landcover_classes.csv" #this works

vegetation_carbon_stock_directory = "Z:\\veg_c_storage_rawdata\\vegetation_carbon_stock_global"

output_path = "Z:\\veg_c_storage_rawdata"


rootdir = os.path.dirname(__file__)

landcover_directory = r"\\akif.internal\public\z_resources\im-wb\landcove_layers"

landcover_classes_csv = os.path.join(rootdir, 'im.nca.postprocessing\aggregation.region.classified\tmp')

def check_os_path(path):
    """
    build the path according the OS running the script
    """
    if platform.system() == "Windows":
        path = path.replace("/","\\")
    else:
        path = path.replace("\\","/")
    return path


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
            address = os.path.join(path, file)
            #build the path according the OS running the script
            address = check_os_path(address)
            if address not in file_list:
                # Add the file address to the list if it had not been added before.
                file_list.append(address)
        else:
            pass
    
    return file_list
# WARNING: both layers must have the same ammount of files corresponding to the same years
landcover_list = get_raster_data(landcover_directory)
vcs_list = get_raster_data(vegetation_carbon_stock_directory)

def load_landcover_classes(file):
    """
    load_landcover_classes loads a csv containing a table describing
    the landcover classes.

    :param file: the address of the csv with the landcover classification.
    :return: a DataFrame with the data classification.
    """
    file = check_os_path(file)
    df = gpd.read_csv(file)
    return df

clc_df = pd.read_csv(landcover_classes_csv)


def create_output_directory(path_location, folder_name):
    """
    Create and go the folder which will contain the outputs
    This method raise FileExistsError if the directory to be created already exists.
    """
    
    dir = os.path.join(path_location, + folder_name)
    dir = check_os_path(dir)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except Exception as e:
            print(e)
            
    print(dir)
    return dir

output_dir = create_output_directory(output_path, "vegetation_carbon_stock_global_reclass")

"""starting of the reading"""
# we take into account that there are as many landcover layers as vsc ones and correspond to the same year periods
def generate_vsc_classes(vcs_list, landcover_list, clc_df, output_dir):
    """This part iterates over the carbon vegetation layers and the landcover ones
    at the same time, classifiing them over a gdal calculation
    :vcs_list: list of vegetation carbon layers
    :landcover_list: list of carbon vegetation layers
    """
    failed_layers = []
    for vsc_file, landcover_file in zip(vcs_list[0:1],landcover_list[0:1]): # vcs_list[0:1],landcover_list[0:1]
        print(vsc_file)
        print(landcover_file)
        """read the landcover csv and select the value and the class"""
        for row_index, row in clc_df.loc[1:1].iterrows(): # clc_df.loc[1:1].iterrows()
            print(row_index)
            
            ## Raster Calculator##
            # Arguements.
            landcover_file
            vsc_file
            raster_output_name = os.path.basename(vsc_file).replace('.tif','') + "_" + str(row['VALUE']) + ".tif" #this takes the dir file
            output_file_path = os.path.join(output_dir, raster_output_name)
            calc_expr = "((A == {0})*1 + (A > {0})*(A < {0})*0)*B".format(row['VALUE'])
            try:
                # NOTE: add the format always
                calculated_tiff = gdal_calc.Calc(calc_expr, A=landcover_file, B=vsc_file, outfile=output_file_path, NoDataValue=0, format="GTiff", creation_options=["COMPRESS=DEFLATE", "TILED=YES"], extent="intersect", quiet=True)
            except Exception as e:
                print(e)
                failed_layers.append(raster_output_name)
            
            print("the layer {0} is finished".format(raster_output_name))
            calculated_tiff = None
            
    print(failed_layers)    
    return print("the process is finished")
  
