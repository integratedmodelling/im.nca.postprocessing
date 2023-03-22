from multiprocessing import Pool

def iteration(x):
    return 0

with Pool as pool:
    
    ymin = 2001
    ymax = 2020
    arguments = [(raster_list_2000, country_polygons, path),(raster_list_2001, country_polygons, path), ]

    result = pool.map(aggregate_density_obs, arguments)