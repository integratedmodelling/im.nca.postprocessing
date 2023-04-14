"""
Module containing the processing functions for the aggregation.
"""

import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import gc
import re
from geoutils import area_of_pixel, get_geometry_grid



def extract_sublist(raster_files_list, value, position):
    """
    extract_sublist extracts a sublist filtering the main one according to the int value we
    want to filter from and its corresponding possition on the string values.

    :param raster_files_list: a list containing the raster files of a directory
    :param value_possition: the possition of the value on the string
    :param position: position of the int value in relation to all in the string
    :return: a filtered list
    """
    sublist_raster_files = []
    for item in raster_files_list:
        # If the value and the corresponding possition value of the list are equal, the path is appended.
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
        # Tuple with the arguments for the execution of one year.
        argument_year = (list_year, region_polygons, temp_export_path)
        # append the tuple to the list
        argument_list.append(argument_year)
        
    return argument_list



def split_and_aggregate(out_image, out_transform, pixel_size, width, height):
    """
    split_and_aggregate splits a raster in tiles of 1000x1000 pixels, performs
    the aggregation in each tile and accumulates the results to obtain the total
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

            tile_agg_value = aggregate_array(out_image, out_transform, pixel_size, w0, h0, w1, h1)

            accumulated_agg_value += tile_agg_value

    return accumulated_agg_value



def aggregate_array(out_image, out_transform, pixel_size, width_0, height_0, width_1, height_1):
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
    ys = cols = rows = None #empty the memory

    # Iterate over the latitudes matrix, calculate the area of each tile, and
    # store it in the real_raster_areas array.
    real_raster_areas = np.empty(np.shape(latitudes))
    for i, latitude_array in enumerate(latitudes):
        for j, latitude in enumerate(latitude_array):
            real_raster_areas[i,j] = area_of_pixel(pixel_size, latitude)

    # Calculate the total value in each tile: density * area = observable value
    # in the area.
    value = real_raster_areas * out_image[0,height_0:height_1,width_0:width_1] #I don't think np.transpose() is necesary
    out_image = None #empty the memory
    # Sum all the carbon stock values in the country treating NaNs as 0.0.
    aggregated_value = np.nansum(value)

    return aggregated_value



def aggregate_density_observable_parallel_wrapper(args):
    return aggregate_density_observable(args[0],args[1],args[2])


def process_region_by_chunks(region_grid, raster_file, pixel_size, row):

     # Start iteration over each tile and accumulate the observable's value.
    total_aggregated_value = 0
    total_tiles = int(len(region_grid))
    for row_index, tile_row in region_grid.iterrows():
        
        print("Processing tile {} out of {} for region {}".format(row_index + 1, total_tiles, row["ADM0_NAME"]))
        
        geo_tile_row = gpd.GeoSeries(tile_row['geometry'])
        
        # Mask the raster file with the corresponding tile.
        out_image, out_transform = rasterio.mask.mask(raster_file, geo_tile_row, crop=True)
        
        # Print memory usage for control.
        print("Tile data memory usage: {} Gb".format(out_image.nbytes / np.power(10.0,9))) 

        # Obtain the number of tiles in both directions.
        height = out_image.shape[1]
        width  = out_image.shape[2]

        # Calculate the aggregated value of the observable within the tile.
        aggregated_value = aggregate_array(out_image, out_transform, pixel_size, 0, 0, width, height)
        
        total_aggregated_value += aggregated_value 
        
        # Clean memory.
        out_image = None 
        out_transform = None
        gc.collect()

    return total_aggregated_value    


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

    for file in raster_files_list[:]: 
       
        # Iterate over all the raster files' addresses and extract the year from the filename.
        file_year = re.findall(r'\d+', file)[0]
        try:
            strata = re.findall(r'\d+', file)[2]
        except:
            strata = False

        print("Processing file {} corresponding to year {}.".format(file, file_year))

        # This list will store the results from the aggregation.
        aggregated_value_list = []

        with rasterio.open(file) as raster_file: # Load the raster file.

            gt = raster_file.transform # Get all the raster properties on a list.
            pixel_size = gt[0] # X size is stored in position 0, Y size is stored in position 4.

            error_regions_id = [] # Create a list collecting all regions processed with errors.
            
            for row_index, row in region_polygons.iterrows(): 
                
                try:
                    # Iterate over the region polygons to progressively calculate the total carbon stock in each one of them. Each region is used to mask the raster with the observable data. Each region is processed in chunks of predefined size to avoid live memory overflow. Assuming the worst case scenario of data type being float64 (8bytes), an array of 10000x10000 weights around 800Mb. PCs RAM must at least equal 800Mb*nCPUSWorking. A larger margin is advisable to avoid killing the process.   
                    raster_size = (100000,100000)    

                    # Extracting the country's polygon geometry in a GeoDataFrame.
                    geodf_row = gpd.GeoDataFrame(geometry=gpd.GeoSeries(row['geometry']), crs=4326) 
                        
                    # Create a grid of the territory with the coresponding EPSG and desired raster size within each grid tile.
                    grid = get_geometry_grid(geodf_row, "EPSG:4326", raster_size)

                    # Adjust the grid to the shape of the territory. This operation requires both inputs to be GDFs.
                    region_grid = grid.overlay(geodf_row, how="intersection").to_crs(epsg='4326') 

                    # print("region_grid memory usage {}".format(sys.getsizeof(region_grid)))
                    
                    # Start iteration over each tile and accumulate the observable's value.
                    aggregated_value = process_region_by_chunks(region_grid,raster_file,pixel_size,row)
                    
                except Exception as e:
                    # In case there is an error on the process, a value of -9999.0 will be appended.
                    print("Region {} with index {} exited with errors: {}. Assigning aggregated value to -9999.0 in final dataset.".format(row["ADM0_NAME"], row["OBJECTID"], e) )
                    error_regions_id.append(row["OBJECTID"])
                    aggregated_value = -9999.0 

                # Add the aggregated stock to the list.
                aggregated_value_list.append(aggregated_value)

                print("Processing finished for region {} with index {}. Observable's aggregated value: {}".format(row["ADM0_NAME"], row["OBJECTID"], aggregated_value))
        
        print("Year {} finished.".format(file_year))
        print("ID of regions processed with errors: ", error_regions_id)

        # Transform the list to a DataFrame using the year as header.
        if not strata:
            aggregated_observable = pd.DataFrame(aggregated_value_list, columns = [file_year])
            # Export the temporary results from curent year.
            aggregated_observable.to_csv(temp_export_path + "_" + str(file_year) + ".csv")
        else:
            aggregated_observable = pd.DataFrame(aggregated_value_list, columns = [file_year + "_" + strata])
            aggregated_observable.to_csv(temp_export_path + "_" + str(file_year) + "_" + str(strata) + ".csv")
            

        # Merge this year's results with the final, multi-year DataFrame.
        aggregated_df = pd.merge(aggregated_df, aggregated_observable, how='outer', left_index = True, right_index=True)

    aggregated_df.to_csv(temp_export_path + "_total_" + str(file_year) + ".csv")

    return aggregated_df