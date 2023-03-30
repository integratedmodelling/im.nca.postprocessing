"""
Module containing the Input/Output functions needed to run geospatial aggregation scripts.
"""

import os
import geopandas as gpd
import pandas as pd
import platform


def get_raster_data(path):
    """
    get_raster_data gets the addresses of all the raster files ("*.tif") stored in the directory specified by "path". Each raster file should correspond to a different year for the analysis.

    :param path: directory containing the raster files for the density observable to aggregate.
    :return: a list storing the addresses of all the raster files.
    """

    file_list = []
    for file in os.listdir(path):
        # Iterate over all the files in the specified directory.
        if ".tif" in file:
            # Process the file if it has a .tif format.
            if platform.system() == "Windows":
                address = os.path.join(path, file).replace("/","\\")
            else:
                address = os.path.join(path, file).replace("\\","/")
                #build the path according the OS running the script

            if address not in file_list:
                # Add the file address to the list if it had not been added before.
                file_list.append(address)
        else:
            pass

    return file_list


def load_region_polygons(file):
    """
    load_region_polygons loads a shapefile containing a polygon layer describing the regions to perform the aggregation.

    :param file: the address of the shapefile with the region polygons.
    :return: a GeoDataFrame with the data on the regions' polygons.
    """

    # Build the path according the OS running the script.
    if platform.system() is "Windows":
        file = file.replace("/","\\")
    else:
        file = file.replace("\\","/")

    gdf = gpd.read_file(file)

    return gdf


def export_to_csv(region_polygons, aggregated_observable, path):
    """
    export_to_csv joins the result of the aggregation process with the regions and exports the final dataset in CSV format to the specified path.

    :param region_polygons: a GeoDataFrame storing the polygons corresponding to each region used to aggregate.
    :param aggregated_observable: a DataFrame storing the aggregated values of the observable.
    :param path: path for the export.
    :return: None. The function creates a file in the specified path with the final results of the aggregation.
    """

    # Create a DataFrame based on the regions GeoDataFrame and dropping unnecessary information in order to keep only: the polygons' Id, region codes, and administrative names.

    df_final = pd.DataFrame(region_polygons.drop(columns='geometry'))
    
    df_final = df_final.drop(["STATUS", "DISP_AREA", "ADM0_CODE", "STR0_YEAR", "EXP0_YEAR", "Shape_Leng", "ISO3166_1_", "ISO3166__1", "Shape_Le_1", "Shape_Area"], axis = 1)

    # Join the depurated regions DataFrame with the aggregated values.
    df_final = df_final.join(aggregated_observable)

    # Export the result to the specified path.
    df_final.to_csv(path)



