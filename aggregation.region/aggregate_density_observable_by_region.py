"""
This script performs the aggregation by region of density observables (i.e.
quantity per unit area). Given a series of raster files with maps of the density
observables and a polygon layer specifying the regions for the aggregation, the
script iterates over the rasters and aggregates the observable at the region
level. The script treats each raster file as maps of the same observable for
different years. The raster filenames are expected to have the following
structure "vcs_YYYY_global_300m.tif" so the script can extract the year
associated with the map.

The aggregation of a density observable requires knowledge of the area of each
raster tile. The effect of latitude on tile area is accounted for.

The script produces temporary outputs for every raster file (e.g. year) in the
form of a CSV file with two columns: one for the ID of the region and the other
for the value of the aggregated observable in that region. The final output of
the script is a table in CSV format containing the aggregated values for each
region and each year.

The script has the following structure:

1- Specify case-dependent variables for the aggregation process.
   THIS IS THE ONLY SECTION THE FINAL USER IS CONCERNED WITH AND THAT NEEDS TO
   BE UPDATED AT EACH USE IN FUNCTION OF THE APPLICATION DOMAIN.

    These are:
    - directory of the raster.
    - region polygons shapefile.
    - name of the observable to aggregate.
    - export directories.

2- Declaration of functions.
3- Aggregation process.

Date: 01/11/2022
Authors: Rubén Crespo, Diego Bengochea Paz.
"""

import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import math
import platform
import re
from multiprocessing import Pool

###############################################################################
# Specify variables for the current aggregation process.
###############################################################################

# The path to the directory containing the raster files with the data on the
# density observable to aggregate. # Both Windows and Unix types of path writing
# are supported.
# Note that the raster filenames must have the following structure:
# vcs_YYYY_global_300m.tif.
# TODO: This will be changed in the future for generality.
# raster_directory = r"\\akif.internal\public\veg_c_storage_rawdata\vegetation_carbon_stock_global"
raster_directory = r"Z:\veg_c_storage_rawdata\vegetation_carbon_stock_global"

# Path to the shapefile containing the data on region polygons that are to be
# used in the aggregation.
# region_polygons_file = r"\\akif.internal\public\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"
region_polygons_file = r"Z:\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"

# Name of the observable to aggregate. This will be used as a prefix for the
# temporary results filenames. A suffix "_YYYY.csv" will be appended to this
# name. The directory for the temporary exports is defined below.
observable_name = "vegetation-carbon-stock"

# Path for the temporal exports of the aggregation process after each raster
# processed.
temp_export_dir = "./vegetation.carbon.stock/tmp/vcs.aggregated.country/"

# Path to export the final dataset.
export_path = "./vegetation.carbon.stock/vcs-aggregated-country.csv"

###############################################################################
###############################################################################


# WARNING: THE CODE BELOW SHOULD NOT BE MODIFIED BY THE FINAL USER !


###############################################################################
# Begin of functions' declaration.
###############################################################################

###############################################################################
# Input/Output functions.
###############################################################################


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
    load_region_polygons loads a shapefile containing a polygon layer describing
    the regions to perform the aggregation.

    :param file: the address of the shapefile with the region polygons.
    :return: a GeoDataFrame with the data on the regions' polygons.
    """
    if platform.system() is "Windows":
        file = file.replace("/","\\")
    else:
        file = file.replace("\\","/")
        #build the path according the OS running the script

    gdf = gpd.read_file(file)
    return gdf


def export_to_csv(region_polygons, aggregated_observable, path):
    """
    export_to_csv joins the result of the aggregation process for each year with
    the regions and exports the final dataset in CSV format to the specified
    path.

    :param region_polygons: a GeoDataFrame storing the polygons corresponding to
    each region used to aggregate.
    :param aggregated_observable: a DataFrame storing the aggregated values of
    the observable.
    :param path: path for the export.
    :return: None. The function creates a file in the specified path with the
    final results of the aggregation.
    """

    # Create a DataFrame based on the regions GeoDataFrame and dropping
    # unnecessary information in order to keep only: the polygons' Id, region
    # codes, and administrative names.
    df_final = pd.DataFrame(region_polygons.drop(columns='geometry'))
    df_final = df_final.drop(["STATUS", "DISP_AREA", "ADM0_CODE", "STR0_YEAR", "EXP0_YEAR", "Shape_Leng", "ISO3166_1_", "ISO3166__1", "Shape_Le_1", "Shape_Area"], axis = 1)

    # Join the depurated regions DataFrame with the aggregated values.
    df_final = df_final.join(aggregated_observable)

    # Export the result to the specified path.
    df_final.to_csv(path)


###############################################################################
# Processing function.
###############################################################################


def area_of_pixel(pixel_size, center_lat):
    """
    area_of_pixel calculates the area, in hectares, of a wgs84 square raster
    tile given its latitude and side-length.
    This function is adapted from https://gis.stackexchange.com/a/288034.

    :param pixel_size: is the length of the pixel side in degrees.
    :param center_lat: is the latitude of the center of the pixel. This value
    +/- half the `pixel-size` must not exceed 90/-90 degrees latitude or an
    invalid area will be calculated.
    :return: the rel area in hectares of a square pixel of side length
    `pixel_size` whose center is at latitude `center_lat`.
    """

    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*math.sin(math.radians(f))
        zp = 1 + e*math.sin(math.radians(f))
        area_list.append(
            math.pi * b**2 * (
                math.log(zp/zm) / (2*e) +
                math.sin(math.radians(f)) / (zp*zm)))
    return (pixel_size / 360. * (area_list[0] - area_list[1])) * np.power(10.0,-4)


def split_and_aggregate(out_image, out_transform, pixel_size, width, height):
    """
    split_and_aggregate splits a raster in tiles of 1000x1000 pixels, performs
    the aggregation in each tile and accumulate the results to obtain the total
    value for all the region. The function is called when the raster mask
    corresponding to the region is too large in size (>3Gb).

    :param out_image: is the masked raster layer.
    :param out_transform: the Affine containing the transformation matrix with
    latitude and longitude values, resolution, etc.
    :param pixel_size: is the side length in degrees of each square raster tile.
    :param width: is the width of the masked layer.
    :param height: is the height of the masked layer.
    :return: the value of the aggregated observable at the region-level.
    """

    tilesize = 1000
    # The variable to accumulate the aggregated value of the observable.
    accumulated_agg_value = 0

    for i in range(0, width, tilesize): # Tilesize marks from where to where in width.
        for j in range(0, height, tilesize):
            # This is for the edge parts, so we don't get nodata from the borders.
            w0 = i # Start of the array.
            w_plus = min(i+tilesize, width) - i # Addition value.
            w1 = w0 + w_plus # End of the array.
            h0 = j
            h_plus = min(j+tilesize, height) - j
            h1 = h0 + h_plus

            tile_agg_value = aggregate_one_region(out_image, out_transform, pixel_size, w0, h0, w1, h1)

            accumulated_agg_value += tile_agg_value

    return accumulated_agg_value

def aggregate_one_region(out_image, out_transform, pixel_size, width_0, height_0, width_1, height_1):
    """
    aggregate_one_region performs the aggregation of the density observable in a
    given region supplied in the form of a masked raster of the observable.

    :param out_image: the masked raster layer to aggregate.
    :param out_transform: the Affine of the raster layer.
    :param pixel_size: the side length in degrees of each square raster tile.
    :param width_0: the starting value position of the width array
    :param height_0: the starting value position of the height array
    :param width_1: the end value position of the width array
    :param height_1: the end value position of the height array
    :return: the aggregated value of the observable in the specified region.
    """

    # Create a matrix of coordinates based on tile number.
    cols, rows = np.meshgrid(np.arange(width_0, width_1), np.arange(height_0, height_1))

    # Transform the tile number coordinates to real coordinates and extract only
    # latitude information.
    ys = rasterio.transform.xy(out_transform, rows, cols)[1] # [0] is xs
    latitudes = np.array(ys) # Cast the list of arrays to a 2D array for computational convenience.

    # Iterate over the latitudes matrix, calculate the area of each tile, and
    # store it in the real_raster_areas array.
    real_raster_areas = np.empty(np.shape(latitudes))
    for i, latitude_array in enumerate(latitudes):
        for j, latitude in enumerate(latitude_array):
            real_raster_areas[i,j] = area_of_pixel(pixel_size, latitude)

    # Calculate the total value in each tile: density * area = observable value
    # in the area.
    value = real_raster_areas * out_image[0,height_0:height_1,width_0:width_1] #I don't think np.transpose() is necesary

    # Sum all the carbon stock values in the country treating NaNs as 0.0.
    aggregated_value = np.nansum(value)

    return aggregated_value

def extract_sublist(raster_files_list, value, position):
    """
    extract_sublist extracts a sublist filtering the main one according to the int value we
    want to filter from and its corresponding possition on the string values.
    :raster_files_list: a list containing the raster files of a directory
    :value_possition: the possition of the value on the string
    :position:
    """
    sublist_raster_files = []
    for item in raster_files_list:
        # if the value and the corresponding possition value of the list are equal, the path is appended
        if value == int(re.findall(r'\d+', item)[position]):
            sublist_raster_files.append(item)
        else:
            continue
        
    return sublist_raster_files

def parallel_argument_list(year_list, raster_files_list, region_polygons, temp_export_path):
    """
    parallel_argument_list creates a list containing different tupples from different input parameters
    :param year_list: the list of years to operate
    :param raster_files_list: a list containing the raster files of a directory
    :param region_polygons: a GeoDataFrame storing the polygons corresponding to
    each region used for the aggregation.
    :param temp_export_path: path to export the temporary results.
    :return: a DataFrame storing the aggregated vegetation carbon stocks at the
    region level for each year.
    
    """
    argument_list = []
    
    for year in year_list:
        list_year = extract_sublist(raster_files_list, year, 0)
        # we create a tuple
        argument_year = (list_year, region_polygons, temp_export_path)
        # append the tuple to the list
        argument_list.append(argument_year)
        
    return argument_list

def aggregate_density_observable(raster_files_list, region_polygons, temp_export_path):
    """
    aggregate_density_observable aggregates the density observable for all the
    raster files specified and inside the specified regions. The result of the
    aggregation is returned as a table and the result for each year/raster is
    progressively exported in CSV format.

    :param raster_files_list: a list containing the addresses of all the raster
    files that store the observable's data for each year.
    :param region_polygons: a GeoDataFrame storing the polygons corresponding to
    each region used for the aggregation.
    :param temp_export_path: path to export the temporary results.
    :return: a DataFrame storing the aggregated vegetation carbon stocks at the
    region level for each year.
    """

    # Final DataFrame will store the aggregated carbon stocks for each country and each year.
    aggregated_df = pd.DataFrame([])

    for file in raster_files_list[:]: # [10:]
        # Iterate over all the raster files' addresses and extract the year from the address.
        file_year = re.findall(r'\d+', file)[0]
        try:
           landcover_class = re.findall(r'\d+', file)[2]
        except:
            landcover_class = False

        print("Processing file {} corresponding to year {}.".format(file, file_year))

        # This list will store the results from the aggregation.
        aggregated_value_list = []

        with rasterio.open(file) as raster_file: # Load the raster file.

            gt = raster_file.transform # Get all the raster properties on a list.
            pixel_size = gt[0] # X size is stored in position 0, Y size is stored in position 4.

            error_countries_id = [] # Create a list for all encountered possible errors
            for row_index, row in region_polygons.iterrows(): # gdf.loc[0:1].iterrows(): / gdf.loc(axis=0)[0:1] / df[df['column'].isin([1,2])]
                try:
                    # Iterate over the country polygons to progressively calculate the total carbon stock in each one of them.

                    geo_row = gpd.GeoSeries(row['geometry']) # This is the country's polygon geometry.

                    # Masks the raster over the current country. The masking requires two outputs:
                    # out_image: the array of the masked image. [z, y, x]
                    # out_transform: the Affine containing the transformation matrix with lat / long values, resolution...
                    out_image, out_transform = rasterio.mask.mask(raster_file, geo_row, crop=True)

                    # Obtain the number of tiles in both directions.
                    height = out_image.shape[1]
                    width  = out_image.shape[2]

                    # Split the masked raster if it is too large to avoid memory
                    # errors. Else process the entire region.
                    if out_image.nbytes > (3* 10**9):
                        print("Country {} exceeds 3Gb of memory, splitting the array in tiles of 1000x1000. Current size is GB: {} .".format(row["ADM0_NAME"], (out_image.nbytes) / np.power(10.0,9)))
                        aggregated_value = split_and_aggregate(out_image, out_transform, pixel_size, width, height)

                    else:
                        aggregated_value = aggregate_one_region(out_image, out_transform, pixel_size, 0, 0, width, height)

                except Exception as e:
                    # In case there is an error on the process, a value of -9999.0 will be appended
                    print("the country {} with index {} has errors: {}".format(row["ADM0_NAME"], row["OBJECTID"], e) )
                    error_countries_id.append(row["OBJECTID"])
                    aggregated_value = -9999.0 

                # Add the aggregated stock to the list.
                aggregated_value_list.append(aggregated_value)

                print("the country {} with index {} is finished with total carbon of: {}".format(row["ADM0_NAME"], row["OBJECTID"], aggregated_value))

        print("Finished calculating year {}.".format(file_year))
        print("countries id with error: ", error_countries_id)

        # Transform the list to a DataFrame using the year as header.
        if not landcover_class:
            aggregated_observable = pd.DataFrame(aggregated_value_list, columns = [file_year])
            # Export the temporary results from curent year.
            aggregated_observable.to_csv(temp_export_path + "_" + str(file_year) + ".csv")
        else:
            aggregated_observable = pd.DataFrame(aggregated_value_list, columns = [file_year + "_" + landcover_class])
            aggregated_observable.to_csv(temp_export_path + "_" + str(file_year) + "_" + str(landcover_class) + ".csv")
            

        # Merge this year's results with the final, multi-year DataFrame.
        aggregated_df = pd.merge(aggregated_df, aggregated_observable, how='outer', left_index = True, right_index=True)

    aggregated_df.to_csv(temp_export_path + "_total_" + str(file_year) + ".csv")

    return aggregated_df

###############################################################################
# End of functions' declaration.
###############################################################################

###############################################################################
# Begin the aggregation process.
###############################################################################
if __name__ == "__main__":
    print("Loading data.")
    raster_list = get_raster_data(raster_directory)
    region_polygons = load_region_polygons(region_polygons_file)
    print("Data loaded succesfully.")
    
    # Full path for temporary exports.
    temp_export_path = temp_export_dir + observable_name
    
    print("Starting aggregation process.")
    #list of years to compute
    year_list = range(2000,2004)
    argument_list = parallel_argument_list(year_list, raster_list, region_polygons, temp_export_path)
    with Pool as pool:
        result = pool.map(aggregate_density_observable,argument_list)
        print(result)
        
    print("Aggregation finished.")
    # aggregated_observable = aggregate_density_observable(raster_list, region_polygons, temp_export_path)


    print("Exporting the aggregated dataset.")
    # export_to_csv(region_polygons, result, export_path)
    print("Done.")