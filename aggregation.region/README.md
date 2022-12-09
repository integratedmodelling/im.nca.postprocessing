# Aggregation of density variables by region

***aggregate_densities_by_region.py*** processes the maps of density variables produced with ARIES in order to obtain region-level data. The _Python_ script has a twin _notebook_ that serves as interactive documentation.   

## Outline of the aggregation process

1) Load the global raster files produced with ARIES.
2) Load the vector file containing the information on regional polygons.
3) Iterate over the maps and over regions.
4) For each map and region: iterate over every raster tile belonging to the region, calculate total value of the observable accounting for pixel area and sum the result.
5) Progressively store the results in a list and produce a final table that is exported in CSV format.  

## Domains of application

- ***Vegetation carbon stock***: the ARIES model, based on state-of-the-art IPCC methodology, produces global carbon stock maps at a resolution of 300 meters for each year between 2001 and 2020. Each raster tile contains information on the vegetation carbon stock per hectare at the location. The analysis takes into account national borders and tile area to aggregate carbon stocks at country level, producing, as a result, a ***dataset on total vegetation carbon stocks per country for the whole world and for the last two decades***. The dataset is stored in ***vegetation.carbon.stock*** together with a _Python_ script to produce visualization of the vegetation carbon stock dataset. 
