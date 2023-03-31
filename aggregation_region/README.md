# Aggregation of a density observable by region

## Modules

Three modules are provided that implement methods to perform the spatial aggregation of density observables by user-specified regions:

1) _geoio.py_ provides a series of handy input/output methods specifically designed to import geospatial data and export summary CSV tables with the results from the aggregation procedure, in a convenient way for the algorithm that is used. 
2) _geoutils.py_ provides a series of utility methods for geospatial analysis, like the estimation of a raster pixel real area from its coordinates and size.
3) _processing.py_ provides all the methods needed to process the input data and calculate the observable's aggregated value in each region of the analysis. For multi-year analysis the method used for aggregation supports multithreading, assigning one CPU per year of the analysis. 

## Directory structure

The directory *notebooks* contains a series of Jupyter notebooks implementing the aggregation algorithms. They serve the purpose of interactive documentation. 

The rest of directories are dedicated to specific applications of the aggregation procedure. Within each directory there are specific scripts using the module's function for the aggregation of the observable. In the case of the vegetation carbon stock data, a script is provided to perform the aggregation at the country-level, both with the overall value and stratified by landcover (more detail below). The aggregation process is however identical, only the input data is different. The script _aggregate_by_country.py_ can be replicated in other applications and serve thus as documentation.

## Outline of the aggregation process

1) Load the raster files (tipically corresponding to multiple years) with the observable's data produced with ARIES.
2) Load the vector file containing the information on regional polygons.
3) Iterate over the maps and over regions.
4) For each map and region: iterate over every raster tile belonging to the region, calculate total value of the observable in that tile accounting for tile area and sum the result.
5) Progressively store the results and produce a final table that is exported in CSV format.  

## Domains of application

- ***Vegetation carbon stock***: the ARIES model, based on state-of-the-art IPCC methodology, produces global carbon stock maps at a resolution of 300 meters for each year between 2001 and 2020. Each raster tile contains information on the vegetation carbon stock per hectare at the location. The analysis takes into account national borders and tile area to aggregate carbon stocks at country level, producing, as a result, a ***dataset on total vegetation carbon stocks per country for the whole world and for the last two decades***. The dataset is stored in CSV format inside ***vegetation.carbon.stock*** directory together with a _Python_ script to produce visualization of the vegetation carbon stock dataset. 
