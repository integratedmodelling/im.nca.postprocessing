# Aggregation of a density observable by region

***aggregate_density_observable_by_region.py*** processes maps of density observables produced with ARIES in order to obtain region-level data. The _Python_ script has a twin _notebook_ that serves as interactive documentation.   

Data obtained by the application of the aggregation method to specific domains is stored in different directories within this one. Further details on the deomains of application can be found below.

## Outline of the aggregation process

1) Load the raster files produced with ARIES.
2) Load the vector file containing the information on regional polygons.
3) Iterate over the maps and over regions.
4) For each map and region: iterate over every raster tile belonging to the region, calculate total value of the observable in that tile accounting for tile area and sum the result.
5) Progressively store the results and produce a final table that is exported in CSV format.  

## Domains of application

- ***Vegetation carbon stock***: the ARIES model, based on state-of-the-art IPCC methodology, produces global carbon stock maps at a resolution of 300 meters for each year between 2001 and 2020. Each raster tile contains information on the vegetation carbon stock per hectare at the location. The analysis takes into account national borders and tile area to aggregate carbon stocks at country level, producing, as a result, a ***dataset on total vegetation carbon stocks per country for the whole world and for the last two decades***. The dataset is stored in CSV format inside ***vegetation.carbon.stock*** directory together with a _Python_ script to produce visualization of the vegetation carbon stock dataset. 
