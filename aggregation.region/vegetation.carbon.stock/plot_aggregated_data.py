import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import math
import platform
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap

def get_vcs_filenames(path):
    """
    Store the filenames of the vegetation carbon stock data for every year in a
    list.
    :param path: is the path to the data directory.
    :return: a list containing all the filenames.
    """
    file_list = []
    for file in os.listdir(path):
        # Iterate over all the files in the specified directory.
        if ".csv" in file:
            # Process the file if it has a .tif format.
            if platform.system() == "Windows":
                address = os.path.join(path, file).replace("/","\\")
            else:
                address = os.path.join(path, file).replace("\\","/")
                #build the path according the OS running the script

            if address not in file_list:
                # Add the file address to the list if it had not been added before.
                file_list.append(address)
        else:
            pass
    return file_list

def merge_vcs_all_years(vcs_files):
    """
    Merges all the vegetation carbon stock data for every year in a single
    DataFrame with the years as headers.
    :param vcs_files: a list with the paths to every vegetation carbon stock CSV
    data file.
    :return: a DataFrame with all the data merged and indexed by country index.
    """

    # Iterate over the filenames to progressively load and merge them.
    for file in vcs_files:

        # Load the file
        vcs = pd.read_csv(file)
        vcs = vcs.rename( columns={'Unnamed: 0' : "cid" } )
        try:
            vcs_df = pd.merge(vcs_df, vcs, on = ["cid"])
        except:
            vcs_df = pd.DataFrame(vcs)

    return vcs_df[sorted(vcs_df)]

def load_countries_polygon_data(countries_file):
    """
    Loads the countries' polygon data in a GeoDataFrame and removes the unneeded
    columns.
    :param countries_file: is the polygon layer with countries names and
    polygons.
    :return: the reduced countries GeoDataFrame with one column with the names
    and a second one with the geometry.
    """

    countries_polygons = gpd.read_file(countries_file)
    countries_polygons = countries_polygons[["OBJECTID","ADM0_NAME","geometry"]]
    countries_polygons = countries_polygons.rename( columns = {"OBJECTID":"cid", "ADM0_NAME":"name"})
    countries_polygons["cid"] = (countries_polygons["cid"] - 1).astype(int)
    return countries_polygons

def join_vcs_with_country(vcs_df, countries_gdf):
    """
    Joins the vegetation carbon stock DataFrame with the country names and
    polygons.
    :param vcs_df: is the DataFrame storing the vegetation carbon stock data.
    :param countries_gdf: is a GeoDataFrame containing the country names and
    geometries.
    :return: a GeoDataFrame with country names and polygons and the associated
    vegetation carbon stock in tonnes.
    """
    joined = vcs_df.merge(countries_gdf, on="cid")
    joined = joined.drop("cid",axis=1)
    # print(joined)
    return joined

def vcs_differences(gdf, init_year, last_year, time_interval):
    """
    Calculates the differences in vegetation carbon stocks between evenly spaced
    years for every country.
    :param gdf: is the country-vegetation carbon stock dataset.
    :param init_year: is the initial year of the analysis.
    :param last_year: is the last year of the analyis.
    :param time_interval: is the time step for the analysis.
    :return: a GeoDataframe with data on the differences in vegetation carbon
    stock between the given years.
    """

    init_years = np.arange(init_year,last_year + 1 ,time_interval)[:-1].tolist()
    last_years  = np.arange(init_year,last_year + 1 ,time_interval)[1:].tolist()
    years = list(zip(init_years, last_years))

    diff_gdf = gdf[["name","geometry"]]
    for y0,y1 in years:
        diff_gdf[str(y0)+"-"+str(y1)] = (gdf.loc[:,str(y1)]-gdf.loc[:,str(y0)])

    return diff_gdf

def vcs_relative_differences(gdf, init_year, last_year, time_interval):
    """
    Calculates the differences in vegetation carbon stocks between evenly spaced
    years for every country.
    :param gdf: is the country-vegetation carbon stock dataset.
    :param init_year: is the initial year of the analysis.
    :param last_year: is the last year of the analyis.
    :param time_interval: is the time step for the analysis.
    :return: a GeoDataframe with data on the differences in vegetation carbon
    stock between the given years.
    """

    init_years = np.arange(init_year,last_year + 1 ,time_interval)[:-1].tolist()
    last_years  = np.arange(init_year,last_year + 1 ,time_interval)[1:].tolist()
    years = list(zip(init_years, last_years))

    diff_gdf = gdf[["name","geometry"]]
    for y0,y1 in years:
        diff_gdf[str(y0)+"-"+str(y1)] = 100*(gdf.loc[:,str(y1)]-gdf.loc[:,str(y0)])/gdf.loc[:,str(y0)]

    return diff_gdf

def get_winners_and_losers(gdf, n, init_year, last_year):
    """
    Gets the top n winner/loser countries in terms of changes in vegetation
    carbon stock between two years.
    :param gdf: the dataset.
    :param init_year: the initial year.
    :param last_year: the last year.
    :return: a duple of Series with the names of the countries in the top 10 of
    winners and losers regarding vegetation carbon stock changes.
    """

    diff = vcs_differences(gdf, init_year, last_year, last_year - init_year)

    winners = diff.nlargest(n,[str(init_year)+"-"+str(last_year)]).name
    losers  = diff.nsmallest(n,[str(init_year)+"-"+str(last_year)]).name

    return (winners,losers)

def get_relative_winners_and_losers(gdf, n, init_year, last_year):
    """
    Gets the top n winner/loser countries in terms of relative changes in
    vegetation carbon stock between two years.
    :param gdf: the dataset.
    :param init_year: the initial year.
    :param last_year: the last year.
    :return: a duple of Series with the names of the countries in the top 10 of
    winners and losers regarding vegetation carbon stock changes.
    """

    diff = vcs_relative_differences(gdf, init_year, last_year, last_year - init_year)

    winners = diff.nlargest(n,[str(init_year)+"-"+str(last_year)]).name
    losers  = diff.nsmallest(n,[str(init_year)+"-"+str(last_year)]).name

    return (winners,losers)

def plot_vcs_dynamics(gdf, countries, type, title):
    """
    Plots the vegetation carbon stock dynamics for all the available years and
    only the specified countries.
    :param gdf: the dataset.
    :param countries: is a Series with the country names.
    :return: a plot of the vegetation carbon dynamics.
    """

    # Restrict the dataset to the specified countries.
    gdf = gdf[ gdf.name.isin(countries) ]

    # Drop the geometry column.
    gdf = gdf.drop(columns=["geometry"])

    # Change the unit to megatonnes.
    list_of_years = ["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"]
    gdf[list_of_years] = gdf[list_of_years]/np.power(10,6)

    # Tidy the dataframe.
    gdf = pd.melt(gdf, id_vars="name", var_name="year", value_name="vcs")
    gdf = gdf.rename(columns={"name": "Country"})

    # fig, ax = plt.subplots(1, 1)

    # Create the figure.
    sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    palette = sns.color_palette("pastel")
    g = sns.relplot(data=gdf,
                # ax=ax,
                x="year", y="vcs",
                hue="Country",
                kind="line",
                marker='o',
                palette=palette,
                height = 5.0,
                aspect = 1.0
    )
    g.axes[0,0].set_xlabel("Year")
    g.axes[0,0].set_ylabel("Vegetation carbon stock (Mt)")
    sfont = {'fontname':'sans-serif'}
    g.fig.suptitle(title)
    g.axes[0,0].set_yscale("log")
    # g.add_legend()
    # textwrap.fill(l, 20) for l in gdf.columns
    plt.subplots_adjust(top=0.9,left=0.15,bottom=0.1)
    # plt.tight_layout()
    if type == "w":
        plt.savefig("./figures/dynamics/vcs_dynamics_winners.png", dpi=300)
    if type == "l":
        plt.savefig("./figures/dynamics/vcs_dynamics_losers.png", dpi=300)
    # plt.show()
    plt.close()

def plot_relative_vcs_dynamics(gdf, countries, type, title):
    """
    Plots the vegetation carbon stock dynamics for all the available years and
    only the specified countries.
    :param gdf: the dataset.
    :param countries: is a Series with the country names.
    :return: a plot of the vegetation carbon dynamics.
    """

    # Restrict the dataset to the specified countries.
    gdf = gdf[ gdf.name.isin(countries) ]

    # Drop the geometry column.
    gdf = gdf.drop(columns=["geometry"])

    # Calculate relative change with respect to initial year.
    list_of_years = ["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"]
    gdf[list_of_years] = gdf[list_of_years].div(gdf["2001"], axis=0)*100-100

    # Tidy the dataframe.
    gdf = pd.melt(gdf, id_vars="name", var_name="year", value_name="vcs")
    gdf = gdf.rename(columns={"name": "Country"})

    # fig, ax = plt.subplots(1, 1)

    # Create the figure.
    sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    palette = sns.color_palette("pastel")
    g = sns.relplot(data=gdf,
                # ax=ax,
                x="year", y="vcs",
                hue="Country",
                kind="line",
                marker='o',
                palette=palette,
                height = 5.0,
                aspect = 1.0
    )
    g.axes[0,0].set_xlabel("Year")
    g.axes[0,0].set_ylabel("Relative percentual change")
    sfont = {'fontname':'sans-serif'}
    g.fig.suptitle(title)
    plt.subplots_adjust(top=0.9,left=0.15,bottom=0.1)
    # plt.tight_layout()
    if type == "w":
        plt.savefig("./figures/dynamics/relative_vcs_dynamics_winners.png", dpi=300)
    if type == "l":
        plt.savefig("./figures/dynamics/relative_vcs_dynamics_losers.png", dpi=300)
    # plt.show()
    plt.close()


def plot_vcs_map(gdf, year, vcs_range):
    """
    Plots the vegetation carbon stock world maps for the specified year.
    :param gdf: is the dataset.
    :param year: the year to visualize.
    :param vcs_range: is a tuple with the minimum and maximum vegetation carbon
    stock values in the dataset to produce colorbars consistent across every year.
    :return: a global map of the vegetation carbon stocks for the specified year.
    """

    gdf = gpd.GeoDataFrame(gdf[[str(year),"geometry"]])
    gdf[str(year)] = gdf[str(year)]/np.power(10,6)

    fig, ax = plt.subplots(1, 1, figsize=(9,4))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)

    # Need to figure out how to specify colorbar ranges.
    gdf.plot(column=str(year),
             ax=ax,
             legend=True,
             # legend_kwds={'label':"Vegetation carbon stock (tonnes)"},
             cmap = 'summer_r',
             norm=colors.LogNorm(vmin=vcs_range[0], vmax=vcs_range[1]),
             cax=cax
    )
    sfont = {'fontname':'sans-serif'}
    ax.set_title("Vegetation carbon stock (Mt) - " + str(year),**sfont)
    ax.set_axis_off()
    plt.tight_layout(w_pad=0.05, h_pad=0.03)
    # plt.subplots_adjust(top=0.99,left=0.01,bottom=0.01)
    plt.savefig("./figures/maps/vcs_"+str(year)+".png", dpi=300)
    # plt.show()
    plt.close()


def plot_vcs_differences_map(gdf, init_year, last_year):
    """
    Plots the vegetation carbon stock relative differences between two specified
    years.
    :param gdf: is the dataset.
    :param init_year: is the initial year to compute the difference.
    :param last_year: is the last year to compute the difference.
    :return: a world map of the relative difference in carbon stock for every
    country and between the two specified years.
    """

    diff = gpd.GeoDataFrame(vcs_relative_differences(gdf, init_year, last_year, last_year - init_year))
    col = str(init_year)+"-"+str(last_year)

    fig, ax = plt.subplots(1, 1, figsize=(9,4))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)

    palette = sns.diverging_palette(20, 200, s=95, l=70, sep=1, as_cmap=True)

    diff.plot(column=col,
              ax=ax,
              legend=True,
              # legend_kwds={'label':"Relative Change of Vegetation Carbon Stock \n" + str(init_year)+"-"+str(last_year)},
              cmap = palette, #sns.color_palette("coolwarm", as_cmap=True),#sns.color_palette("Spectral", as_cmap=True), #'RdBu',
              norm = colors.TwoSlopeNorm(vmin=diff[col].min(), vcenter=0., vmax=diff[col].max()),
              cax=cax
    )
    sfont = {'fontname':'sans-serif'}
    ax.set_title("Relative change of vegetation carbon stock (%) \n" + str(init_year)+"-"+str(last_year),**sfont)
    ax.set_axis_off()
    plt.tight_layout(w_pad=0.05, h_pad=0.03)
    plt.savefig("./figures/maps/vcs_change_"+ str(init_year)+"_"+str(last_year)+".png", dpi=300)
    # plt.show()
    plt.close()


def plot_carbon_stock_cummulative_distribution(gdf,year):
    """
    Plots the distribution of vegetation carbon stock across countries for a
    specified year.
    :param gdf: is the dataset.
    :param year: is the year of the analysis.
    :return: a figure depicting the distribution of vegetation carbon stocks
    across countries.
    """

    gdf = gdf[[str(year),"geometry"]]
    gdf[str(year)] = gdf[str(year)]/gdf[str(year)].sum()*100

    # fig, ax = plt.subplots(1, 1)
    sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    # palette = sns.color_palette("pastel")
    palette = sns.set_palette("pastel",color_codes=True)
    g = sns.displot(data=gdf,
                # ax=ax,
                y=str(year),
                kind="ecdf",
                palette = palette,
                color = "r",
                height=5,
                aspect=1.5
                # fill=True,
                # cut=0
    )
    g.axes[0,0].set_xlabel("Proportion of countries")
    g.axes[0,0].set_ylabel("Percentage of world's vegetation carbon stock")
    # g.axes[0,0].set_xscale("log")
    g.axes[0,0].set_yscale("log")
    g.fig.suptitle("Cummulative distribution of vegetation carbon stock - " + str(year))
    plt.subplots_adjust(top=0.9)
    plt.savefig("./figures/distributions/vcs_cdf_"+str(year)+".png", dpi=300)
    # plt.show()
    plt.close()


def plot_carbon_stock_distribution(gdf,year):
    """
    Plots the distribution of vegetation carbon stock across countries for a
    specified year.
    :param gdf: is the dataset.
    :param year: is the year of the analysis.
    :return: a figure depicting the distribution of vegetation carbon stocks
    across countries.
    """

    gdf = gdf[[str(year),"geometry"]]
    gdf = gdf.drop( gdf[gdf[str(year)]<10.0].index )
    # gdf[str(year)] = gdf[str(year)]/gdf[str(year)].sum()*100

    # fig, ax = plt.subplots(1, 1)
    sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    # palette = sns.color_palette("pastel")
    palette = sns.set_palette("pastel",color_codes=True)
    g = sns.displot(data=gdf,
                # ax=ax,
                x=str(year),
                kind="hist",
                palette = palette,
                color = "b",
                log_scale = (True,False),
                kde=True,
                height=5,
                aspect=1.5
                # fill=True,
                # cut=0
    )
    g.axes[0,0].set_ylabel("Number of countries")
    g.axes[0,0].set_xlabel("Vegetation carbon stock (tonnes)")
    # g.axes[0,0].set_xscale("log")
    g.axes[0,0].set_yscale("log")
    g.fig.suptitle("Distribution of vegetation carbon stock in tonnes - " + str(year))
    plt.subplots_adjust(top=0.9)
    plt.savefig("./figures/distributions/vcs_df_"+str(year)+".png", dpi=300)
    # plt.show()
    plt.close()


def plot_difference_vs_initial(gdf, init_year, last_year):
    """
    Creates a scatter plot of relative difference in vegetation carbon stock vs.
    the vegetation carbon stock at the first year.
    :param gdf: is the dataset.
    :param init_year: is the initial year to compute the difference and calculate
    the mean.
    :param last_year: is the last year to compute the difference and calculate
    the mean.
    :return: a scatter plot of relative change vs. average vegetation carbon stock.
    """

    diff = vcs_relative_differences(gdf, init_year, last_year, last_year - init_year)
    gdf0 = gdf[[str(init_year),"name"]]
    gdf0[str(init_year)] = gdf0[str(init_year)]/np.power(10,6)

    gdf1 = pd.merge(diff, gdf0, on = ["name"])
    gdf1 = gdf1.drop( gdf1[gdf1[str(init_year)]<10.0].index )

    # fig, ax = plt.subplots(1, 1)

    g = sns.jointplot(data=gdf1,
                  # ax=ax,
                  x=str(init_year),
                  y=str(init_year)+"-"+str(last_year),
                  alpha = .5,
                  s=35,
                  edgecolor=".2",
                  linewidth=.5,
                  marginal_kws=dict(log_scale=(True,False)),
    )
    g.set_axis_labels( xlabel = "Vegetation carbon stock (Mt) in " + str(init_year),
                       ylabel = "Relative change (%) between "+str(init_year)+" and "+str(last_year)
                     )
    g.ax_joint.set_xscale('log')
    g.ax_marg_x.set_xscale('log')
    g.fig.suptitle("Initial carbon stock vs. relative changes.")
    plt.subplots_adjust(top=0.9)
    plt.savefig("./figures/distributions/vcs_init_vs_changes_"+str(init_year)+"_"+str(last_year)+".png", dpi=300)
    # g.axes[0,0].set_ylabel("Relative change in vegetation carbon between "+str(init_year)+" and "+str(last_year))
    # g.axes[0,0].set_xscale("log")
    # plt.show()
    plt.close()

def plot_vcs_10_largest(gdf, year):
    """
    Create a bar plot of the vegetation carbon stock of for the countries with
    the 10 largest carbon stocks in the world and for a given year.
    :param gdf: is the dataset.
    :param year: is the year for the analysis.
    :return: a bar plot.
    """

    # Get the data for the specified year.
    gdf_year = gdf[[str(year),"name"]]

    # Get the ten countries with the largest carbon stock.
    gdf_largest = gdf_year.nlargest(10,str(year))

    # Transform units to Mt
    gdf_largest[str(year)] = gdf_largest[str(year)]/np.power(10,6)

    sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    palette = sns.set_palette("pastel",color_codes=True)

    g = sns.catplot(
        data=gdf_largest, y="name", x=str(year), kind="bar", height=9, #aspect=1.5,
        palette = palette
    )
    g.set_axis_labels("Vegetation Carbon Stock (Mt)", "")
    g.set_yticklabels(rotation=45)
    g.set_yticklabels(textwrap.fill(x.get_text(), 10) for x in g.ax.get_yticklabels())
    g.fig.suptitle("Vegetation carbon stock in the 10 countries with the largest stock")
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.12)
    # plt.show()
    plt.savefig("./figures/distributions/vcs_barplot_top10_"+str(year)+".png", dpi=300)
    plt.close()



# Preliminary data treatment.
file_list = get_vcs_filenames("./temp_data/")
vcs_df = merge_vcs_all_years(file_list)
countries_gdf = load_countries_polygon_data("./country_polygons/2015_gaul_dataset_mod_2015_gaul_dataset_global_countries_1.shp")
gdf = join_vcs_with_country(vcs_df,countries_gdf)

# Figure production.

# Array of years of the analysis.
years = np.arange(2001,2019,1)

# Creation of the vegetation carbon stock maps

# Remove the very small vegetation carbon stocks to allow for better
# visualization. This would not be necessary if plotting carbon stock density.

vcs_threshold = 1000000.0
gdf_reduced = gpd.GeoDataFrame(gdf)
for year in years:
    gdf_reduced = gdf_reduced.drop( gdf_reduced[gdf_reduced[str(year)]<vcs_threshold].index )

# Calculate range for the colorbar. The range is shared by every map to faciltate
# visual comparison between years.
years_str = [str(item) for item in years]
vcs_min = gdf_reduced[years_str].min().min()/np.power(10,6)
vcs_max = gdf_reduced[years_str].max().max()/np.power(10,6)
# print(vcs_min)
# print(vcs_max)

for year in years:
    plot_vcs_map(gdf_reduced,year,(vcs_min,vcs_max))

plot_vcs_differences_map(gdf, 2001, 2005)
plot_vcs_differences_map(gdf, 2005, 2010)
plot_vcs_differences_map(gdf, 2010, 2015)
plot_vcs_differences_map(gdf, 2015, 2018)
plot_vcs_differences_map(gdf, 2001, 2018)
plot_vcs_differences_map(gdf, 2001, 2010)
plot_vcs_differences_map(gdf, 2010, 2018)

# Create dynamics for winners and losers

winners, losers = get_winners_and_losers(gdf,5,2001,2018)

plot_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

winners, losers = get_relative_winners_and_losers(gdf,5,2001,2018)

plot_relative_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_relative_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

winners, losers = get_winners_and_losers(gdf,5,2001,2010)

plot_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

winners, losers = get_relative_winners_and_losers(gdf,5,2001,2010)

plot_relative_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_relative_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

winners, losers = get_winners_and_losers(gdf,5,2010,2018)

plot_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

winners, losers = get_relative_winners_and_losers(gdf,5,2010,2018)

plot_relative_vcs_dynamics(gdf,winners,"w","Vegetation carbon stock dynamics of top 5 winner countries")
plot_relative_vcs_dynamics(gdf,losers,"l","Vegetation carbon stock dynamics of top 5 loser countries")

# Plot distributions
for year in years:
    plot_carbon_stock_distribution(gdf,year)
    plot_carbon_stock_cummulative_distribution(gdf,year)

# Plot change vs. initial vegetation carbon stock.
plot_difference_vs_initial(gdf, 2001, 2018)
plot_difference_vs_initial(gdf, 2001, 2010)
plot_difference_vs_initial(gdf, 2010, 2018)

# Plot top 10 biggest stocks.
for year in years:
 plot_vcs_10_largest(gdf, year)
