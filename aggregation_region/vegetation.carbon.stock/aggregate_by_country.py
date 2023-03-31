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

from aggregation_region.geoio import get_raster_data, load_region_polygons
from multiprocessing import Pool
from aggregation_region.processing import parallel_argument_list, aggregate_density_observable


# The path to the directory containing the raster files with the data on the
# density observable to aggregate. # Both Windows and Unix types of path writing
# are supported.
# Note that the raster filenames must have the following structure:
# vcs_YYYY_global_300m.tif.
# TODO: This will be changed in the future for generality.
raster_directory = r"\\akif.internal\public\veg_c_storage_rawdata\vegetation_carbon_stock_global"
# raster_directory = r"Z:\veg_c_storage_rawdata\vegetation_carbon_stock_global"

# Path to the shapefile containing the data on region polygons that are to be
# used in the aggregation.
country_polygons_file = r"\\akif.internal\public\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"
# region_polygons_file = r"Z:\veg_c_storage_rawdata\wb_global_countries\2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp"


# Name of the observable to aggregate. This will be used as a prefix for the
# temporary results filenames. A suffix "_YYYY.csv" will be appended to this
# name. The directory for the temporary exports is defined below.
observable_name = "vegetation-carbon-stock"

# Path for the temporal exports of the aggregation process after each raster
# processed.
# temp_export_dir = ".\vegetation.carbon.stock\tmp\\"
temp_export_dir = "C:/Users/admin/Documents/01_Ruben_Scripts/im.nca.postprocessing/aggregation.region/vegetation.carbon.stock/tmp/"
# temp_export_dir = ".\\vegetation.carbon.stock\tmp"

# Path to export the final dataset.
export_path = "./tmp/vcs-aggregated-country-landcover.csv"

if __name__ == "__main__":
    print("Loading data.")
    raster_list = get_raster_data(raster_directory)
    region_polygons = load_region_polygons(country_polygons_file)
    print("Data loaded succesfully.")
    
    # Full path for temporary exports.
    temp_export_path = temp_export_dir + observable_name
    
    print("Starting aggregation process.")
    #list of years to compute
    year_list = range(2001,2021) # always +1
    argument_list = parallel_argument_list(year_list, raster_list, region_polygons, temp_export_path)
    with Pool(processes= 1) as pool:
        print("Starting Pool.")
        # result = pool.starmap(aggregate_density_observable,argument_list)
        result = pool.starmap(aggregate_density_observable,argument_list)
        print(result)
        
    print("Aggregation finished.")
    # aggregated_observable = aggregate_density_observable(raster_list, region_polygons, temp_export_path)


    print("Exporting the aggregated dataset.")
    # export_to_csv(region_polygons, result, export_path)
    print("Done.")