import os
from multiprocessing import Pool

import sys
sys.path.insert(1, '../aggregation.region') # this works
# from aggregate_density_observable_by_region import get_raster_data
import aggregate_density_observable_by_region as ado


raster_directory = r"\\akif.internal\public\veg_c_storage_rawdata\vegetation_carbon_stock_landcover_global"
# raster_directory = r"Z:\veg_c_storage_rawdata\vegetation_carbon_stock_landcover_global"

# Path to the shapefile containing the data on region polygons that are to be
# used in the aggregation.
region_polygons_file = r"\\akif.internal\public\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"
# region_polygons_file = r"Z:\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"

# Name of the observable to aggregate. This will be used as a prefix for the
# temporary results filenames. A suffix "_YYYY.csv" will be appended to this
# name. The directory for the temporary exports is defined below.
observable_name = "vegetation-carbon-stock"

# Path for the temporal exports of the aggregation process after each raster
# processed.
temp_export_dir = "./tmp/vcs.aggregated.country/"

# Path to export the final dataset.
export_path = "./vegetation.carbon.stock/vcs-aggregated-country-landcover.csv"

raster_list = ado.get_raster_data(raster_directory)
region_polygons = ado.load_region_polygons(region_polygons_file)
print("Data loaded succesfully.")
# time 14s

# Full path for temporary exports per ladncover and year.
temp_export_path = temp_export_dir + observable_name

print("Starting aggregation process.")
#list of years to compute
year_list = range(2000,2020)
argument_list = ado.parallel_argument_list(year_list, raster_list, region_polygons, temp_export_path)
with Pool() as pool:
    print("starting Pool")
    result = pool.starmap(ado.aggregate_density_observable,argument_list)
    print(result)
        
        
# aggregated_observable = ado.aggregate_density_observable(raster_list, region_polygons, temp_export_path)
print("Aggregation finished.")

#primer error peto por temp_export_dir, asi que ha llegado hasta ahí

print("Exporting the aggregated dataset.")
# ado.export_to_csv(region_polygons, aggregated_observable, export_path)
print("Done.")
print(raster_list[1:2])

