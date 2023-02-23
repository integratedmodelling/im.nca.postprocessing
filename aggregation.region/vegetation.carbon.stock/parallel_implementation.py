from multiprocessing import Pool

def iteration(x):
    return 0

with Pool as pool:
    ymin = 2001
    ymax = 2020
    result = pool.map(iteration, range(ymin,ymax))