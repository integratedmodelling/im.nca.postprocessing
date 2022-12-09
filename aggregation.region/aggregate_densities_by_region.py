"""
This script performs the aggregation of global vegetation carbon stocks in Tonnes per Hectare at the country-level for the 2001-2020 period.
The result is a CSV table storing the total vegetation carbon stock in Tonnes for each country in the entire world and for each year between 2001 and 2020.

The script is structured in the following way: 
- l.19-73:   declaration of the functions used for Input/Output.
- l.73-133:  declaration of the function where the aggregation process is implemented.
- l.136-161: main program where the aggregation process is carried on. 
"""

import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import math
import platform

"""
Begin of functions' declaration.
"""

"""
Input/Output functions.
"""

def get_raster_data(path):
    """
    get_raster_data gets the addresses of all the raster files ("*.tif") contained in the directory specified by the path. Each raster file
                         corresponds to a different year.

    :param path: directory containing the raster data for the global vegetation carbon stocks at 300m resolution for each year.
    :return: a list storing the addresses of all the raster files containing the data to be aggregated by country. 
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
    # print(file_list)
    return file_list

def load_country_polygons(file):
    """
    load_country_polygons loads a shapefile containing vector data of country borders for the entire world as a GeoDataFrame.
    
    :param file: the address of the shapefile with the country border data.
    :return: a GeoDataFrame with the data on the countries' polygons. 
    """
    if platform.system() is "Windows":
        file = file.replace("/","\\")
    else:
        file = file.replace("\\","/")
        #build the path according the OS running the script

    gdf = gpd.read_file(file)
    return gdf

def export_to_csv(country_polygons, aggregated_carbon_stocks):
    """
    export_to_csv creates a DataFrame where aggregated vegetation carbon stocks are associated with each country and exports this data in CSV format. 
    
    :param country_polygons: a GeoDataFrame storing the polygons corresponding to each country for the entire world.
    :param aggregated_carbon_stocks: a DataFrame storing the aggregated carbon stock values to be associated with each country.
    :return: None. The function creates a "total_carbon_test.csv" file in the current working directory that contains the total vegetation carbon stock for each country.
    """
    
    # Create a DataFrame based on the country border GeoDataFrame and dropping unnecessary information to keep only: the polygons' Id, country codes, and administrative names.
    df_final = pd.DataFrame(country_polygons.drop(columns='geometry'))
    df_final = df_final.drop(["STATUS", "DISP_AREA", "ADM0_CODE", "STR0_YEAR", "EXP0_YEAR", "Shape_Leng", "ISO3166_1_", "ISO3166__1", "Shape_Le_1", "Shape_Area"], axis = 1)

    # Join the depurated country DataFrame with the aggregated vegetation carbon stocks to associate each country with its total stock.  
    df_final = df_final.join(aggregated_carbon_stocks)
        
    # Export the result to the current working directory.
    df_final.to_csv("total_carbon.csv")

"""
Processing function.
"""

def area_of_pixel(pixel_size, center_lat):
    """
    area_of_pixel calculates the area, in hectares, of a wgs84 square raster tile. 
                  This function is adapted from https://gis.stackexchange.com/a/288034.
    
    :param pixel_size: is the length of the pixel side in degrees.
    :param center_lat: is the latitude of the center of the pixel. This
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    :return: the area of a square pixel of side length `pixel_size` whose center is at latitude `center_lat` in hectares.
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

def carbon_stock_raster_tiling(out_image, out_transform, pixel_size, width, height):
    """
    raster_tiling is called when the output of the masking raster exceeds the 3Gb storage. To not get memory issues, we split the masked raster in tiles of 1000x1000, and calculate the total carbon stock, accumulating the value of the total carbon for every tile.
    
    :param out_image: is the masked raster layer, in the context of this script the vegetation carbon stock raster.
    :param out_transform: the Affine containing the transformation matrix with latitude and longitude values, resolution...
    :param pixel_size: is the side length in degrees of each square raster tile.
    :param width: is the width of the masked layer.
    :param height: is the height of the masked layer.

    :return: the value of the total carbon stock for the country
    """

    tilesize = 1000
    total_acumulated_carbon_stock = 0

    for i in range(0, width, tilesize): #tilesize marks from where to where in width
        for j in range(0, height, tilesize):
            #this is for the edge parts, so we don't get nodata from the borders
            w0 = i #start of the array
            w_plus = min(i+tilesize, width) - i #addition value
            w1 = w0 + w_plus #end of the array
            h0 = j #start of the array
            h_plus = min(j+tilesize, height) - j #addition value
            h1 = h0 + h_plus #end of the array

            total_carbon_stock = get_total_carbon_stock(out_image, out_transform, pixel_size, w0, h0, w1, h1)

            total_acumulated_carbon_stock += total_carbon_stock

    return total_acumulated_carbon_stock

def get_total_carbon_stock(out_image, out_transform, pixel_size, width_0, height_0, width_1, height_1):
    """
    get_total_carbon_stock creates a raster layer based on a reference layer (out_image), where the value of each tile corresponds to its true area in hectares.
    
    :param out_image: is the baseline raster layer, in the context of this script the vegetation carbon stock raster.
    :param out_transform: the Affine containing the transformation matrix with latitude and longitude values, resolution...
    :param pixel_size: is the side length in degrees of each square raster tile.
    :param width_0: is the starting value position of the width array
    :param height_0: is the starting value position of the height array
    :param width_1: is the end value position of the width array
    :param height_1: is the end value position of the height array
    
    :return: the total carbon stock extracted from the raster.
    """
    
    # Create a matrix of coordinates based on tile number.
    cols, rows = np.meshgrid(np.arange(width_0, width_1), np.arange(height_0, height_1))
    
    # Transform the tile number coordinates to real coordinates and extract only latitude information. 
    ys = rasterio.transform.xy(out_transform, rows, cols)[1] # [0] is xs
    latitudes = np.array(ys) # Cast the list of arrays to a 2D array for computational convenience.

    # Iterate over the latitudes matrix, calculate the area of each tile, and store it in the real_raster_areas array.
    real_raster_areas = np.empty(np.shape(latitudes))
    for i, latitude_array in enumerate(latitudes):
        for j, latitude in enumerate(latitude_array):
            real_raster_areas[i,j] = area_of_pixel(pixel_size, latitude)

    # Calculate the total carbon stock in each tile: tonnes/hectare * hectares = tonnes.    
    total_carbon_stock_array = real_raster_areas * out_image[0,height_0:height_1,width_0:width_1] #I don't think np.transpose() is necesary

    # Sum all the carbon stock values in the country treating NaNs as 0.0. 
    total_carbon_stock = np.nansum(total_carbon_stock_array) 

    return total_carbon_stock

def carbon_stock_aggregation(raster_files_list, country_polygons):
    """
    carbon_stock_aggregation aggregates vegetation carbon stock data in Tonnes per Hectare and with a resolution of 300m at the country level. 
                             The result of the aggregation is the total vegetation carbon stock in Tonnes for each country. Naturally, the 
                             dependence of raster tile area on the latitude is taken into account. The function iterates over the carbon stock 
                             rasters corresponding to different years.
    
    :param raster_files_list: a list containing the addresses of all the raster files that store the vegetation carbon stock data for each year.
    :param country_polygons: a GeoDataFrame storing the polygons corresponding to each country for the entire world.
    :return: a DataFrame storing the aggregated vegetation carbon stocks at the country level for each year.
    """
    
    # Final DataFrame will store the aggregated carbon stocks for each country and each year. 
    aggregated_carbon_stock_df = pd.DataFrame([])
    
    for file in raster_files_list[:]: # [10:]
        # Iterate over all the raster files' addresses and extract the year from the address. 
        filename_length = 24 # This is the number of characters in the raster file name if the convention "vcs_YYYY_global_300m.tif" is followed.
        start = len(file) - filename_length
        year_string_start = file.find("vcs_",start)
        file_year = str( file[ year_string_start + 4 : year_string_start + 8] )
        
        print("We are working with the file {} from the year {}".format(file, file_year))

        aggregated_carbon_stock_list = [] # This list will store the results from the aggregation. 

        with rasterio.open(file) as raster_file: # Load the raster file.

            gt = raster_file.transform # Get all the raster properties on a list.
            pixel_size = gt[0] # X size is stored in position 0, Y size is stored in position 4.

            for row_index, row in country_polygons.iterrows(): # gdf.loc[0:1].iterrows():
                # Iterate over the country polygons to progressively calculate the total carbon stock in each one of them.
                
                geo_row = gpd.GeoSeries(row['geometry']) # This is the country's polygon geometry.

                # Masks the raster over the current country. The masking requires two outputs:
                # out_image: the array of the masked image. [z, y, x]
                # out_transform: the Affine containing the transformation matrix with lat / long values, resolution...
                out_image, out_transform = rasterio.mask.mask(raster_file, geo_row, crop=True) 
                
                # Obtain the number of tiles in both directions.
                height = out_image.shape[1]
                width  = out_image.shape[2]

                #check the size of the raster image
                if out_image.nbytes > (3* 10**9):
                    print("the country {} exceeds 3Gb of memory, we will split the array in tiles of 1000. Current size is GB: {} ".format(row["ADM0_NAME"], (out_image.nbytes) / np.power(10.0,9)))

                    total_carbon_stock = carbon_stock_raster_tiling(out_image, out_transform, pixel_size, width, height)

                else:
                    # Create a global raster where each pixel's value corresponds to its true area in hectares.
                    total_carbon_stock = get_total_carbon_stock(out_image, out_transform, pixel_size, 0, 0, width, height) 
                
                # Add the aggregated stock to the list.
                aggregated_carbon_stock_list.append(total_carbon_stock)  

                print("the country {} is finished".format(row["ADM0_NAME"]))
                
        print("Finished calculating {} year raster".format(file_year))
    
        # Transform the list to a DataFrame using the year as header.
        aggregated_carbon_stock = pd.DataFrame(aggregated_carbon_stock_list, columns = [file_year]) 

        # Merge this year's carbon stocks to the final, multi-year DataFrame.
        aggregated_carbon_stock_df = pd.merge(aggregated_carbon_stock_df, aggregated_carbon_stock, how='outer', left_index = True, right_index=True)

        #export the carbon stock year as a backup 
        aggregated_carbon_stock.to_csv("carbon_stock_{}.csv".format(file_year))

    return aggregated_carbon_stock_df

"""
End of functions' declaration.
"""

"""
Aggregation of vegetation carbon stock at the country level. 
"""

"""
The directory containing the raster files for the global carbon stock data at 300m resolution. This is the data to be aggregated by country.
Note that the raster filenames must have the following structure: vcs_YYYY_global_300m.tif.
"""

vcs_rasters_directory = r"\\akif.internal\public\veg_c_storage_rawdata" # Both Windows and Unix types of path writing are supported. 

"""
Full address of the shapefile containing the data on country borders for the entire world. This determines the country's polygons 
inside which the aggregation of carbon stocks is done. 
"""

country_polygons_file = r"\\akif.internal\public\z_resources\im-wb\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"

print("Loading data.")
vcs_rasters_list = get_raster_data(vcs_rasters_directory) 
country_polygons = load_country_polygons(country_polygons_file) 
print("Data was loaded succesfully.")

print("Starting aggregation process.")
vcs_aggregated   = carbon_stock_aggregation(vcs_rasters_list, country_polygons) 
print("Aggregation of vegetation carbon stocks at the country level finished.")
export_to_csv(country_polygons, vcs_aggregated) 
print("Total vegetation carbon stocks at the country level succesfully exported.")
