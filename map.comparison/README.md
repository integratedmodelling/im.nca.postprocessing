# Spatial statistics for map comparison

***structural_similarity_indices.py*** calculates the Structural Similarity (SSIM) between two rasters. This implementation follows the methodology described in Jones et al. 2016 _Novel application of a quantitative spatial comparison tool to species distribution data_ (https://www.sciencedirect.com/science/article/pii/S1470160X16302990). The script produces and exports four rasters displaying the following spatially explicit comparison metrics:
- Similarity in mean
- Similarity in variance
- Similarity in pattern
- Overall similarity (weighted geometric mean of the 3 metrics above)
as well as the average value of each similarity index as metrics of global similarity.
