# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:57:37 2025

@author: Dylan
"""

### create funtion for general gravity change transect ficgure

#%% ## base code ##

# Import modules
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import contextily as cx
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.plotting import plot_line
from scipy.interpolate import griddata

# Define coordinate systems
epsg_wgs = 4326 # World Geodetic System 1984
epsg_utm = 32613 # UTM Zone 13N


# Read in data
df = pd.read_csv('del_gravstg_oct_nov.csv')
bnd = gpd.read_file('interpolation_extent/interpolation_extent.shp')

# Convert latitude and longitude to point data
df['geometry'] = gpd.points_from_xy(df['Northing'],
                                     df['Easting'],
                                    crs = epsg_wgs)


eastings = df["Easting"]
northings = df["Northing"]


# Reproject shapefile boundary
bnd  = bnd.to_crs("epsg:32613") #WGS84 UTM 13N

# Reproject points (how? need to turn it into a shapefile?)

# Convert DataFrame to GeoDataFrame
df = gpd.GeoDataFrame(df)
df.head()



# Check CRS for both datasets
print(bnd.crs)
print(df.crs)


# Plotting of map boundary and sampled points
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.ticklabel_format(useOffset=False)
bnd.plot(ax=ax, facecolor='w', edgecolor='k')
df.plot(ax=ax, marker='x', facecolor='yellow', aspect='equal')
plt.show()


# Find and delete points outside of the field boundaries
print(df.shape)
for k,row in df.iterrows(): #Prints row without indexed rows
    point = row['geometry']
    if not point.within(bnd['geometry'].iloc[0]):
        df.drop(k, inplace=True) #Drops the Kth row

df.reset_index(inplace=True, drop=True)
print(df.shape)


# Plot again after deleting the points outside of the boundary
# Plotting of map boundary and sampled points

# Make stem plot where the x is easting and y is gravity anomoly
fig, ax = plt.subplots(3,1, figsize=(15,15), dpi=180)
#plt.figure().set_figwidth(200)
fig.set_size_inches(10, 10)
ax[0].ticklabel_format(useOffset=False)
#bnd.plot(ax=ax, facecolor='w', edgecolor='k')
#ax.plot([northings[0]], [eastings[0]])
ax[0].plot([northings[1:].min() ,northings[1:].max()],
 [eastings[1:].median() ,eastings[1:].median()])
#plot_line(line, ax)
df.plot(ax=ax[0], marker='x', facecolor='yellow', aspect='equal')
cx.add_basemap(ax[0], crs=epsg_utm, source=cx.providers.Esri.WorldImagery)#, source=cx.providers.CartoDB.Positron)
ax[0].set_xlabel('Northing')
ax[0].set_ylabel('Easting')
#ax[0].set_title('Gravity anomaly vs Northing')
#gdf.plot()

# Subplot 1 Gravity Anomoly
ax[1].stem(df['Northing'], df['del_grav_oct_nov(microGal)'])
ax[1].plot([northings[1:].min() ,northings[1:].max()],
           [5, 5], color='grey', linestyle='dashed')
ax[1].plot([northings[1:].min() ,northings[1:].max()],
           [-5, -5], color='grey', linestyle='dashed')
ax[1].set_xlabel('Northing')
ax[1].set_ylabel('delg (microGal)')
#ax[1].set_title('Gravity anomaly vs Northing')

# Subplot 2 Bouguer Slab Storage Change
ax[2].stem(df['Northing'], df['del_storage_oct_nov(m)'])
ax[2].plot([northings[1:].min() ,northings[1:].max()],
           [0.12, 0.12], color='grey', linestyle='dashed')
ax[2].plot([northings[1:].min() ,northings[1:].max()],
           [-0.12, -0.12], color='grey', linestyle='dashed')
ax[2].set_xlabel('Northing')
ax[2].set_ylabel('Storage Change (m)')
#ax[2].set_title('Storage Change vs Northing')
fig.suptitle('Main Title')
plt.show()
#%% ## figure function with a True statement to 

###INPUTS###
# csv, bnd_shp, epsg_wgs, epsg_utm, 
def gravity_transect_figure(northings, eastings, grav, epsg_utm, epsg_wgs=4326,bnd_filepath=False, bnd_fig=False):
    """
    

    Parameters
    ----------
    eastings : list
        Eastings of gravity data
    northings : list
        Northings of gravity data
    grav : list
        Gravity data (microGal) at every x,y point
    bnd_filepath : string, optional
        The filepath to the boundary shapefile outside which the gravity data 
        will be cropped. The default is false
    epsg_wgs : int
        EPSG WGS coordinate system. Default is WGS84 4326
    epsg_utm : int
        EPSG UTM coordinate system.
    bnd_fig : bool, optional
        Boolean variable that indicates whether a seperate plot will be printed
        of all points and cropping boundary. Default is False.

    Returns
    -------
    Figure subplot (1,3) of points with satellite imagery, gravity values on a 
    stem plot with x-axis being easting, and stem plot of storage change 
    calculated using Bouguer slab assumption with x-axis being easting. The
    source for the satellite imagery is cx.providers.Esri.WorldImagery. 

    """
    #make list of storage change using bouguier slab approximation
    array_grav = np.array(grav)
    del_storage = list(array_grav/41.9)
    # make pandas dataframe with eastings, northings, grav, and storage change
    df = pd.DataFrame(
        {'Easting': eastings,
         'Northing': northings,
         'gravity': grav,
         'storage_change': del_storage
        })
    
    eastings = df['Easting']
    northings = df['Northing']
    grav = df['gravity']
    
    # get geometry x,y in one object
    df['geometry'] = gpd.points_from_xy(northings,
                                  eastings,
                                  crs = epsg_wgs)
    
    
    
    # clip gravity data and plot boundary shape file if true
    if isinstance(bnd_filepath, str):
        bnd = gpd.read_file(bnd_filepath)
        # reproject boundary to intputted coordiante system
        bnd  = bnd.to_crs("epsg:" + str(epsg_utm)) #WGS84 UTM 13N
    
        #plot optional boundary figure
    
        if bnd_fig:
            fig, ax = plt.subplots(1,1, figsize=(8,8))
            ax.ticklabel_format(useOffset=False)
            bnd.plot(ax=ax, facecolor='w', edgecolor='k')
            df.plot(ax=ax, marker='x', facecolor='yellow', aspect='equal')
            plt.show()
            
            # Find and delete points outside of the field boundaries
            print(df.shape)
            for k,row in df.iterrows(): #Prints row without indexed rows
                point = row['geometry']
                if not point.within(bnd['geometry'].iloc[0]):
                    df.drop(k, inplace=True) #Drops the Kth row
        
            df.reset_index(inplace=True, drop=True)
            print(df.shape)
        
    # Make stem plot where the x is easting and y is gravity anomoly
    fig, ax = plt.subplots(3,1, figsize=(15,15), dpi=180)
    fig.set_size_inches(10, 10)
    
    ### Subplot 0: points, stallite imagery, and transect line
    ax[0].ticklabel_format(useOffset=False)
    ax[0].plot([northings[1:].min() ,northings[1:].max()],
     [eastings[1:].median() ,eastings[1:].median()])
    #df.plot(ax=ax[0], marker='x', colormap='viridis')
    ax[0].scatter(df['Northing'], df['Easting'], marker='x', c='yellow')
    cx.add_basemap(ax[0], crs=epsg_utm, source=cx.providers.Esri.WorldImagery)
    ax[0].set_xlabel('Northing')
    ax[0].set_ylabel('Easting')
    #ax[0].set_title('Gravity anomaly vs Northing')
    #gdf.plot()

    ### Subplot 1 Gravity Anomoly
    ax[1].stem(df['Northing'], df['gravity'])
    ax[1].plot([northings[1:].min() ,northings[1:].max()],
               [5, 5], color='grey', linestyle='dashed')
    ax[1].plot([northings[1:].min() ,northings[1:].max()],
               [-5, -5], color='grey', linestyle='dashed')
    ax[1].set_xlabel('Northing')
    ax[1].set_ylabel('delg (microGal)')

    ### Subplot 2 Bouguer Slab Storage Change
    ax[2].stem(df['Northing'], df['storage_change'])
    ax[2].plot([northings[1:].min() ,northings[1:].max()],
               [0.12, 0.12], color='grey', linestyle='dashed')
    ax[2].plot([northings[1:].min() ,northings[1:].max()],
               [-0.12, -0.12], color='grey', linestyle='dashed')
    ax[2].set_xlabel('Northing')
    ax[2].set_ylabel('Storage Change (m)')
  
    fig.suptitle('Main Title')
    plt.show()
    
#define a main function to test the function
#define a main function to test the function
def main():
    # Define arbitrary data for easting, northing, and gravity
    northings = [326036.646, 325922.558, 325861.595, 325881.865,
                 326036.08, 326054.247, 326078.388]
    eastings = [3777102.853, 3777115.305, 3777107.745, 3777110.684,
                3777120.663, 3777122.708, 3777131.56]
    grav = [-9.46999990940094, 14.9800000190735, 5.54000008106232, 
            5.21000003814697, -11.7799999713898, -14.5499999523163,
            -6.44000005722046]  

    # Define EPSG codes (replace with appropriate values for your data)
    epsg_utm = 32613  # Example: WGS 84 / UTM zone 13N
    epsg_wgs = 4326

    # Call the gravity_transect_figure function
    gravity_transect_figure(northings, eastings, grav, epsg_utm, epsg_wgs)


if __name__ == "__main__":
    main()





