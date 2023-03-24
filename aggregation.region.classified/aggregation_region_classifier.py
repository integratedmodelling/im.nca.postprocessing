"""
This script performs the aggregation by region and landcover class of density observables (i.e.
quantity per unit area). Given a series of raster files with maps of the density
observables and a polygon layer specifying the regions for the aggregation, the
script iterates over the rasters and aggregates the observable at the region
level. The script treats each raster file as maps of the same observable for
different years. The raster filenames are expected to have the following
structure "vcs_YYYY_global_300m.tif" so the script can extract the year
associated with the map.

Also it is important that the required data for each year is covered as 
landcover data and the raster files.

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
from multiprocessing import Pool


import sys
# print(sys.executable)
# sys.path.insert(1, '../aggregation.region') # this works only in the jupyternotebook
# from aggregate_density_observable_by_region import get_raster_data
import aggregate_density_observable_by_region as ado #


raster_directory = "/home/ubuntu/vcs_wb/by_landcover/vegetation_carbon_stock_landcover_global"
# raster_directory = r"\\akif.internal\public\veg_c_storage_rawdata\vegetation_carbon_stock_landcover_global"
# raster_directory = r"Z:\veg_c_storage_rawdata\vegetation_carbon_stock_landcover_global"

# Path to the shapefile containing the data on region polygons that are to be used in the aggregation.
region_polygons_file = "/home/ubuntu/vcs_wb/by_landcover/2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"
# region_polygons_file = r"\\akif.internal\public\veg_c_storage_rawdata\wb_global_countries\countries_to_test.shp"
# region_polygons_file = r"Z:\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"

# Name of the observable to aggregate. This will be used as a prefix for the
# temporary results filenames. A suffix "_LCclass_YYYY.csv" will be appended to this
# name. The directory for the temporary exports is defined below.
observable_name = "vegetation-carbon-stock"

# Path for the temporal exports of the aggregation process after each raster
# processed.
temp_export_dir = "/home/ubuntu/vcs_wb/by_landcover/tmp/"
# temp_export_dir = "./tmp/vcs.aggregated.country/"

# Path to export the final dataset.
export_path = "/home/ubuntu/vcs_wb/by_landcover/tmp/vcs-aggregated-country.csv"
# export_path = "./tmp/vcs-aggregated-country-landcover.csv"

raster_list = ado.get_raster_data(raster_directory)
region_polygons = ado.load_region_polygons(region_polygons_file)
print("Data loaded succesfully.")
# time 14s

# Full path for temporary exports per landcover and year.
temp_export_path = temp_export_dir + observable_name

print("Starting aggregation process.")
#list of years to compute
year_list = range(2001,2011) # always + 1
argument_list = ado.parallel_argument_list(year_list, raster_list, region_polygons, temp_export_path)

from contextlib import closing #this will close the Pool

with closing(Pool(processes= 9)) as pool:
    print("Starting Pool.")
    # result = pool.starmap(ado.aggregate_density_observable,argument_list)
    # result = pool.imap_unordered(ado.aggregate_density_observable_parallel_wrapper,argument_list,chunksize=1)

    result = pool.starmap(ado.aggregate_density_observable,argument_list)
    print(result)
        

        
# aggregated_observable = ado.aggregate_density_observable(raster_list, region_polygons, temp_export_path)
print("Aggregation finished.")

print("Exporting the aggregated dataset.")
#ado.export_to_csv(region_polygons, aggregated_observable, export_path)

print("Done.")
# print(raster_list[1:2])

