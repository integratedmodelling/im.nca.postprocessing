"""
Comparison of two raster maps through the Structural Similarity (SSIM) indices described 
in Jones et al. 2016 "Novel application of a quantitative spatial comparison tool to 
species distribution data" (https://www.sciencedirect.com/science/article/pii/S1470160X16302990).

Map similarity is quantified by comparing moving window's statistics of the two maps. The 
method produces local measures of similarity for each pixel, thus allowing spatially-explicit 
comparison between maps. Global metrics of similarity are obtained by averaging over the local
ones.

The script produces 4 comparison maps depicting: 
- Similarity in mean
- Similarity in variance 
- Similarity in pattern
- Overall structural similarity (weighted geometric mean of the 3 metrics above)
and also a CSV file with the global similarity indices.

Script inputs are: 
- Path to both rasters to compare
- Window size for local statistics calculation
- Weights of similarities in mean, variance and pattern for structural similarity calculation
- Filename suffix for the maps and CSV files to be produced.

The script has the following structure: 
1- Specification of script inputs.
2- Functions' declaration.
3- Calculation of the similarity.

Date: 20/01/2023
Author: Diego Bengochea Paz.
"""

import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

import numpy as np
import pandas as pd



# Paths to the two rasters to compare 

raster1 = "/home/dibepa/git/global-above-ground-biomass-ml/agb_predicted_realval.tif"
raster2 = "/home/dibepa/git/global-above-ground-biomass-ml/global_forest_watch_10km/resampled_5m_10km_global_adjusted_manual.tif"

# Window size

window_size = 11

# Weights for the SSIM
alpha = 1
beta = 1
gamma = 1

# Filenames 
filename_suffix = "agbd_globalml_gfw_wsize_"+str(window_size)
# filename_suffix = "agbd_globalml_gfw"

# Begin of function's declaration.

###
# Similarity metrics.
###

def similarity_in_mean(mean1,mean2,val_range):
    """
    :param mean1: the window mean of raster 1.
    :param mean2: the window mean of raster 2.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in mean.
    """

    k1 = 0.01
    c1 = np.power(k1*val_range,2)

    return (2*mean1*mean2+c1)/(mean1**2+mean2**2+c1) 

def similarity_in_variance(var1,var2,val_range):
    """
    :param var1: the window variance of raster 1.
    :param var2: the window variance of raster 2.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in variance.
    """

    k2 = 0.03
    c2 = np.power(k2*val_range,2)

    return (2*np.sqrt(var1)*np.sqrt(var2)+c2)/(var1+var2+c2) 


def similarity_in_pattern(var1,var2,cov,val_range):
    """
    :param var1: the window variance of raster 1.
    :param var2: the window variance of raster 2.
    :param cov: the covariance between both raster windows.
    :param range: the range of values in the window for both rasters.
    :return: the window's similarity in pattern.
    """

    k2 = 0.03
    c3 = 0.5*np.power(k2*val_range,2)

    return (cov+c3)/(np.sqrt(var1)*np.sqrt(var2)+c3) 

###
# Window statistics: mean, variance and covariance.
###

def window_mean(array, weights):
    """
    :param array: the numpy array corresponding to the raster.
    :param weights: the array of weights for the arithmetic mean calculation.
    :return: the weighted mean value of the window.
    """

    return np.sum(weights*array)


def window_variance(array, mean, weights):
    """
    :param array: the numpy array corresponding to the raster.
    :param weights: the array of weights for the arithmetic mean calculation.
    :param mean: the window mean.
    :return: the weighted variance of the window.
    """

    diff = (array - mean)#/mean # re-scaled to the mean to allow comparisons between maps with values at different scales
    return np.sum( weights * np.power(diff,2) )

def window_covariance(array1,array2,mean1,mean2,weights): 
    """
    :param array1: the numpy array corresponding to the raster 1.
    :param array2: the numpy array corresponding to the raster 2.
    :param mean1: the window mean of raster 1.
    :param mean2: the window mean of raster 2.
    :param weights: the array of weights for the arithmetic mean calculation.
    :return: the weighted covariance between rasters in the window.
    """       

    diff1 = array1-mean1
    diff2 = array2-mean2

    return np.sum(weights * diff1 * diff2)

###
# Window creation. 
###

def window_reflection(raster,row,col,d_height,d_width,side_length):
    """
    Creates a window for a cell at the raster's boundary using reflective boundary conditions. 
    Instead of specifying case-specific reflection functions depending on the raster boundary
    (i.e. top, bottom, left, right) the function uses matrix rotations to treat each boundary 
    in the same way. Every window is treated as located at the bottom boundary, thus depending 
    on the real boundary, n 90 degree rotations are executed on the window to treat it as a
    window at the bottom boundary. After reflecting the matrix, the window is rotated back 
    again to its original position.
    :param raster: the raster to select the window from. 
    :param row: the row of the window's central pixel.
    :param col: the column of the window's central pixel.
    :param d_height: the central pixel's distance to the top/bottom borders.
    :param d_width: the central pixel's distance to the left/right borders.
    :param side_length: the window's side_length.
    :return: the numpy array with the reflected window.
    """

    # Half of the window's side_length: used to determine the number of rows/columns to reflect.
    l = np.floor(side_length*0.5)

    # Number of rows and columns to reflect.
    drows = int(np.max([l - d_height,0]))
    dcols = int(np.max([l - d_width,0]) )

    # Height and width of the preliminary window to create before completing it with a reflection.
    height = side_length - drows
    width = side_length - dcols    
    
    # Row and column origins of the preliminary window. The only special case to be treated is when
    # the central pixel is close to the left or top boundary.
    row_origin = int(np.max( [row - l, 0] ))
    col_origin = int(np.max( [col - l, 0] ))
    
    # Create the preliminary window given its origin and shape.
    window = create_window(raster,col_origin,row_origin,width,height)    

    # The preliminary window must be rotated to ensure that it can be treated as a window at the 
    # bottom boundary. The number of 90 degree clockwise rotations to perform depends on the real
    # boundary. Horizaontal and vertical boundaries are treated separately.     

    # By default it is assumed that the window has the right orientation or it is not at a horizontal boundary. 
    # krow is the number of 90 degree rotations.
    krow = 0

    # If the central pixel is close to the top boundary then perform a 180 degree rotation: 2 X 90 degree rotations.
    if row - l < 0:
        krow = 2     

    # By default it is assumed that the window is not at a vertical boundary.
    kcol = 0
    # If the central pixel is close to a vertical boundary... 
    if dcols>0:
        # If the central pixel is close to the left boundary then perform a 270 degree rotation.
        if col-l<0:
            kcol=3
        # Else perform a 90 degree rotation.    
        else: 
            kcol=1         

    # Rotate the window.
    wrot = np.rot90(window,k = krow)
    # Take last drows and put them at the beginning.
    reflection = np.append(wrot,wrot[:drows,:],axis=0)
    # Rotate back.
    window = np.rot90(reflection,k = 4-krow)

    # Rotate the window.
    wrot = np.rot90(window,k = kcol)
    # Take last dcols and put them at the beginning
    reflection = np.append(wrot,wrot[:dcols,:],axis=0)
    # Rotate back
    window = np.rot90(reflection,k = 4-kcol)
  
    return window

def distance_to_border(size,pos):
    """
    Calculates the distance of a row or column to its closest boundary.
    :param size: the size of the raster along the desired dimension (rows or columns).
    :param pos: the row or column index.
    :return: the distance to the closest boundary.
    """
    return np.min([pos,size-pos-1])

def create_window(raster,col,row,width,height):
    """
    Creates a raster window.
    :param col: the window's origin along the columns' axis.
    :param row: the window's origin along the rows' axis.
    :param width: the windows width.
    :param height: the windows height.
    :return: a numpy array with the window.
    """
    return raster.read(1, window=Window(col,row,width,height))
    
def create_weights_window(side_length):
    """
    Creates an array of weights for the window statistics calculation. The array
    has the shape of the window and the weights have gaussian density. Standard
    deviation of the gaussian weights is fixed for the integral over the window to
    be equal to 1 (Following Jones et al. 2016, Novel application of a quantitative 
    spatial comparison tool to species distribution data).
    :param side_length: the side of the window.
    :return: gaussian weights centered at the center of the window. 
    """
    weights = np.ones((side_length,side_length))
    std = weights.size*0.3333333
    return gaussian_filter(weights,sigma=std)

def value_range(array1,array2):
    """
    Calculates the range of values of two arrays combined.
    :param array1: the first array.
    :param array2: the second array.
    :return: the maximum value range of the two arrays.   
    """
    max = np.max(np.array([array1,array2]))
    min = np.min(np.array([array1,array2]))
    return max-min

###
# Window similarities
###

def window_sim(window1,window2,val_range,weights):
    """
    Calculates the similarity in mean of two windows.
    :param window1: the window from first raster.
    :param window2: the window from second raster.
    :param val_range: the value range in the window.
    :param weights: the weights for the window.
    :return: the window's similiarity in mean.
    """
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    return similarity_in_mean(mean1,mean2,val_range)

def window_siv(window1,window2,val_range,weights):
    """
    Calculates the similarity in variance of two windows.
    :param window1: the window from first raster.
    :param window2: the window from second raster.
    :param val_range: the value range in the window.
    :param weights: the weights for the window.
    :return: the window's similiarity in variance.
    """
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    var1 = window_variance(window1,mean1,weights)
    var2 = window_variance(window2,mean2,weights)
    return similarity_in_variance(var1,var2,val_range)

def window_sip(window1,window2,val_range,weights):
    """
    Calculates the similarity in pattern of two windows.
    :param window1: the window from first raster.
    :param window2: the window from second raster.
    :param val_range: the value range in the window.
    :param weights: the weights for the window.
    :return: the window's similiarity in pattern.
    """
    mean1 = window_mean(window1,weights)
    mean2 = window_mean(window2,weights)
    cov = window_covariance(window1,window2,mean1,mean2,weights)
    var1 = window_variance(window1,mean1,weights)
    var2 = window_variance(window2,mean2,weights)
    return similarity_in_pattern(var1,var2,cov,val_range)

###    
# Full-map comparison.
###

def map_index(raster1,raster2,side_length,index):
    """
    Calculates the map for a specified SSIM index.
    :param raster1: a raster to compare.
    :param raster2: the raster to compare with.
    :param side_length: the side length of the moving window.
    :param index: the function for the type of index to calculate.
    :return: a numpy array with the SSIM map for the specified index.  
    """

    # Weights are the same for every window.
    weights = create_weights_window(side_length)

    # Half of the window's side_length: used to determine whether a pixel is close to a boundary.
    l = np.floor(side_length*0.5)

    # Open rasters.
    with rasterio.open(raster1) as src1:
        with rasterio.open(raster2) as src2:

            # Extract raster's size. TODO: add something to ensure that rasters have same dimensions.
            nrows = src1.height
            ncols = src1.width
   
            # Create an empty map of correct dimensions.
            index_map = np.zeros((nrows,ncols))

            # Iterate over each pixel and calculate the index value.
            for row in range(nrows):

                drow = distance_to_border(nrows,row)
                
                for col in range(ncols):

                    dcol = distance_to_border(ncols,col)

                    # If the pixel is close to a boundary then use the reflection method to generate the window. 
                    if (drow < l) or (dcol < l):
                        window1 = window_reflection(src1,row,col,drow,dcol,side_length)
                        window2 = window_reflection(src2,row,col,drow,dcol,side_length)
                    # Else just create a window.    
                    else: 
                        window1 = create_window(src1,col-l,row-l,side_length,side_length)
                        window2 = create_window(src2,col-l,row-l,side_length,side_length)
                    
                    val_range = value_range(window1,window2)

                    metric = index(window1,window2,val_range,weights)

                    index_map[row,col] = metric

    return index_map    

def map_sim(raster1,raster2,side_length):
    """
    Calculate the similarity in mean map.
    :param raster1: a raster to compare.
    :param raster2: the raster to compare with.
    :param side_length: the side length of the moving window.
    :return: a numpy array with the similarity in mean map. 
    """
    return map_index(raster1,raster2,side_length,window_sim)

def map_siv(raster1,raster2,side_length):
    """
    Calculate the similarity in variance map.
    :param raster1: a raster to compare.
    :param raster2: the raster to compare with.
    :param side_length: the side length of the moving window.
    :return: a numpy array with the similarity in variance map. 
    """
    return map_index(raster1,raster2,side_length,window_siv)       

def map_sip(raster1,raster2,side_length):
    """
    Calculate the similarity in pattern map.
    :param raster1: a raster to compare.
    :param raster2: the raster to compare with.
    :param side_length: the side length of the moving window.
    :return: a numpy array with the similarity in pattern map. 
    """
    return map_index(raster1,raster2,side_length,window_sip)       

def map_ssim(sim,siv,sip,alpha,beta,gamma):
    """
    Calculate the similarity in mean map.
    :param sim: the similarity in mean numpy array.
    :param siv: the similarity in variance numpy array.
    :param sip: the similarity in pattern numpy array.
    :param alpha: SIM weight.
    :param beta: SIV weight.
    :param gamma: SIP weigth. 
    :return: a numpy array with the overall structural similarity. 
    """

    # The overall structural similarity is a weighted mean of the other indices. 
    return np.power(sim,alpha)*np.power(siv,beta)*np.power(sip,gamma)
   
###
# Raster creation and export.
###

def export_raster(map, raster_path, filename):
    """
    Creates and exports a raster from a numpy array and based on another raster.
    :param map: the numpy array containing the data on the map to generate.
    :raster_path: the path to the base raster.
    :filename: the exported raster's path.
    :return: None.
    """
    
    with rasterio.Env():

        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = rasterio.open(raster_path).profile

        # And then change the band count to 1, set the
        # dtype to float32, and specify LZW compression.
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')

        with rasterio.open(filename+'.tif', 'w', **profile) as dst:
            dst.write(map.astype(rasterio.float32), 1)

# End of function's declaration.

###
# Structural similarity calculation. 
###

print("Starting structural similarity calculation.")
print("Calculating similarity in mean.")
sim = map_sim(raster1,raster2,window_size)
sim_mean = np.mean(sim)
print("Exporting raster.")
export_raster(sim,raster1,"sim_"+filename_suffix)

print("Calculating similarity in variance.")
siv = map_siv(raster1,raster2,window_size)
siv_mean = np.mean(siv)
print("Exporting raster.")
export_raster(siv,raster1,"siv_"+filename_suffix)

print("Calculating similarity in pattern.")
sip = map_sip(raster1,raster2,window_size)
sip_mean = np.mean(sip)
print("Exporting raster.")
export_raster(sip,raster1,"sip_"+filename_suffix)

print("Calculating overall similarity.")
ssim = map_ssim(sim,siv,sip,alpha,beta,gamma)
ssim_mean = np.mean(ssim)
print("Exporting raster.")
export_raster(ssim,raster1,"ssim_"+filename_suffix)

print("Exporting global indices.")
# The global means of each index are exported to a csv file.
pd.DataFrame(np.array([[sim_mean,siv_mean,sip_mean,ssim_mean]]), columns = ['SIM','SIV','SIP','SSIM']).to_csv("ssim_summary_"+filename_suffix)

            
