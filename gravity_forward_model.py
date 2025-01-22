# -*- coding: utf-8 -*-
"""
Created on Fri May 24 08:36:59 2024

@author: 13055
"""

import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import pygmt
import verde as vd
import pandas as pd
import harmonica as hm

def sign(num):
    return -1 if num < 0 else 1

def gravity_forward_model(filepath, x_range_interp, x_range_grav, coords, porosity, vz_thickness):
    """This function will generate the gravity response of some groundwater
    storage change.
    
    INPUTS
    ======
    filepath (str) is the path to the excel file. The excel file should have two
    sheets named 'highstage' and 'lowstage', each with two columns 'x' and 'f(x)'.
    
    x_range_interp (list) is a list [min, max] of the range of data of the low
    stage over which the interpolation will operate.
    
    x_range_grav (list) is a list [min, max] over which the gravity response
    will be computed.
    
    coords (list) is a list of arrays [easting, northing, upward] in meters
    
    vz_thickness (float) is the thickness of the vadose zone. This is used to calculate
    the vertical distance of the meter from the phreatic surface
    
    porosity (float) is the porosity of the aquifer. This will control the volume
    of water drained per head drop.
    
    OUTPUTS
    =======
    gravity_output (list) is a list od gravitational acceleration values calculated
    at the indicated points."""
    
    
    xls = pd.ExcelFile(filepath)
    df_highstage_full = pd.read_excel(xls, 'highstage', index_col= False)
    df_lowstage_full = pd.read_excel(xls, 'lowstage', index_col= False)
    df_highstage = df_highstage_full.query(f'x <= {x_range_interp[1]} and x >= {x_range_interp[0]}')
    df_lowstage = df_lowstage_full.query(f'x <= {x_range_interp[1]} and x >= {x_range_interp[0]}')

    
    #Create region
    west_boundary = -(df_lowstage['x'].max() - df_lowstage['x'].min())
    east_boundary = df_lowstage['x'].max() - df_lowstage['x'].min()
    north_boundary = df_lowstage['x'].max()
    south_boundary = df_lowstage['x'].min()
    region = (west_boundary, east_boundary, south_boundary, north_boundary)
    
    #create x-axis from range and 5 m increment
  
    # x_axis_min = df_lowstage['x'].min()
    # integer_increment = 0
    # while x_axis_min%5 != 0:
    #     integer_increment =+ 1
    #     x_axis_min = x_axis_min - integer_increment * sign(x_axis_min) 
    #     print(x_axis_min)
    
    # x_axis_max = df_lowstage['x'].max()
    # integer_increment = 0
    # while x_axis_max%5 != 0:
    #     integer_increment =+ 1
    #     x_axis_max = x_axis_max - integer_increment * sign(x_axis_max) 
    #    print(x_axis_max)
        
    x_axis_min, x_axis_max = x_range_grav
    x_axis = np.arange(x_axis_min, x_axis_max, 5)
    x_elements = len(x_axis)
    y_elements = 2 * len(x_axis)
    shape = (x_elements, y_elements) #use to pre-allocate memory
    
    (easting, northing) = vd.grid_coordinates(region=region, shape=shape)
    
    #interpolate both functions using scipy.interpolate.interp1d
    highstage_interp = interp.interp1d(df_highstage['x'], df_highstage['f(x)'], axis=0, kind='nearest')
    lowstage_interp = interp.interp1d(df_lowstage['x'], df_lowstage['f(x)'], axis=0, kind='nearest')
    
    #assign surface and reference
    surface = []
    n = 0
    while n < shape[1]:
        surface.append(np.hstack(highstage_interp(x_axis)))
        n += 1
    surface = np.transpose(np.array(surface))


    reference = []
    m = 0
    while m < shape[1]:
        reference.append(np.hstack(lowstage_interp(x_axis)))
        m += 1
    reference = np.transpose(np.array(reference))
    
    #establish prisms
    density = 1000.0 * np.ones_like(surface)*porosity
    prisms = hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=reference,
        properties={"density": density},
    )
    
    #compute gravity field
    coordinates = vd.grid_coordinates(region, shape=shape, extra_coords= (vz_thickness))
    gravity = prisms.prism_layer.gravity(coordinates, field="g_z")
    grid = vd.make_xarray_grid(
        coordinates, gravity, data_names="gravity", extra_coords_names="extra"
    )
    #plot gravity field
    fig = pygmt.Figure()

    title = "Gravitational acceleration of a layer of prisms"

    with pygmt.config(FONT_TITLE="14p"):
        fig.grdimage(
            region=region,
            projection="X10c/10c",
            grid=grid.gravity,
            frame=["a", f"+t{title}", 'x+l"easting (m)"', 'y+l"northing (m)"'],
            cmap="viridis",
        )

    fig.colorbar(cmap=True, position="JMR", frame=["xf0.01", "x+lmGal"])

    fig.show()
    
    #plot 3d prism
    
    # Create the figure and axes object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(easting, northing, surface, color='blue', alpha=0.5)

    # Plot the reference
    ax.plot_surface(easting, northing, reference, color='red', alpha=0.5)
    
    #plot coords
    
    ax.scatter(coords[0], coords[1], coords[2], color='yellow')

    # Set labels and title
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('Prism Layer')
   
    #compute gravity values at selected points
    select_gravity_points = prisms.prism_layer.gravity(coords, field="g_z") #field specifies downward gravitational acceleration
    
    return select_gravity_points, highstage_interp, lowstage_interp

    #output gravity anomoly at each coordinate
    #tgt_y = np.zeros_like(coords)
    #gravity_output = grid.gravity #.sel(easting=coords, northing=tgt_y, method="nearest")
     
    #return gravity_output

#%%

#import highstage and lowstage data

filepath = 'datasets/phreatic_surface/analyticalmodel_phreatic_surface.xlsx'

###PARAMETERS###
K_real = 1.0 #hydraulic conductivity (m/day)
b = 50 #aquifer thickness (m)
n = 0.3 #porosity
vz_thickness = 0 #vertical distance from phreatic surface (m)
Q1 = 0.3 #highstage volumetric discharge (m^2/d)
Q2 = 0.14 #lowstage volumetric discharge (m^2/d)
x_range = [-150, 150]
x_range_interp = [-130, 130]

###SAMPLING COORDINATES###
#grav_loc_10m = np.array([-45, -35, -25, 25, 35, 45])
smpl_coords_upward = np.array(np.arange(53, 54, 0.1))
smpl_coords_easting = np.array(np.full(shape = len(smpl_coords_upward), fill_value = 24, dtype = np.int64))
smpl_coords_northing = np.array(np.full(shape = len(smpl_coords_upward), fill_value = 2, dtype = np.int64))

coords = [smpl_coords_easting, smpl_coords_northing, smpl_coords_upward]

#compute gravitational acceleration at selected points

gravity_output, highstage_interp, lowstage_interp = gravity_forward_model(filepath, x_range, x_range_interp, coords, n, vz_thickness)

print(gravity_output * (1000/41.9))
#%%
"""
#gravity_forward_model(filepath, x_range, x_range_interp, grav_loc_10m, n, vz_thickness)

gravity_output, highstage_interp, lowstage_interp = gravity_forward_model(filepath, x_range, x_range_interp, grav_loc_10m, n, vz_thickness)

#gravity_along_y0 = gravity_output[26]
grav_loc_10m = grav_loc_10m + 52
selected_gravity_points_10m = gravity_output[grav_loc_10m]

#get head measurements from both surfaces (since the highstage will be used in interp)
well_loc_10m = [-50, -40, -30, -20, 20, 30, 40, 50]
head_highstage_10m = highstage_interp(well_loc_10m)
head_lowstage_10m = lowstage_interp(well_loc_10m)

print("head_highstage_10m", head_highstage_10m)
print("head_lowstage_10m", head_lowstage_10m)
print("grav_point", selected_gravity_points_10m)

"""
