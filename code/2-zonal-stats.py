# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:24:08 2021

@author: Grady

This script calculates the 2000 population of each ethnic boundary identified in Murdock (1959). LandScan data is used.
"""

from rasterstats import zonal_stats
import os
import pandas as pd

#%% 
# Set the working directory to the parent folder of this script 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')

#%%
# Population 2014
# Calculate the population in each polygon. Pixels whose centroid is contained in the polygon are included.

stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/landscan/africa2014.tif', stats=['sum'], geojson_out=True)
stats = [d['properties'] for d in stats]
df = pd.DataFrame(stats)
df.rename(columns={'sum': 'population_2014'}, inplace=True)

#%%
# Population 2015
# Calculate the population in each polygon. Pixels whose centroid is contained in the polygon are included.

stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/landscan/africa2015.tif', stats=['sum'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'sum']]
temp.rename(columns={'sum': 'population_2015'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%%
# Population 2016
# Calculate the population in each polygon. Pixels whose centroid is contained in the polygon are included.

stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/landscan/africa2016.tif', stats=['sum'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'sum']]
temp.rename(columns={'sum': 'population_2016'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%%
# Population 2017
# Calculate the population in each polygon. Pixels whose centroid is contained in the polygon are included.

stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/landscan/africa2017.tif', stats=['sum'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'sum']]
temp.rename(columns={'sum': 'population_2017'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%%
# Population 2018
# Calculate the population in each polygon. Pixels whose centroid is contained in the polygon are included.

stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/landscan/africa2018.tif', stats=['sum'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'sum']]
temp.rename(columns={'sum': 'population_2018'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%% Malaria - temperature suitability for P. falciparum
stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/malaria/2010_TempSuitability.Pf.Index.1k.global_Decompressed.geotiff', stats=['mean'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'mean']]
temp.rename(columns={'mean': 'malaria-pf'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%% Malaria - temperature suitability for P. vivax 
stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/malaria/2010_TempSuitability.Pv.Index.1k.global_Decompressed.geotiff', stats=['mean'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'mean']]
temp.rename(columns={'mean': 'malaria-pv'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%% Africa - Land Surface Forms
stats = zonal_stats('data/final_shp/slave_trade.shp', 'D:/africa_landsurface_forms/Africa_Land_Surface_Downsample_Majority.tif', stats=['majority'], geojson_out=True)
stats = [d['properties'] for d in stats]
temp = pd.DataFrame(stats)
temp = temp[['old_ethnic', 'majority']]
temp.rename(columns={'majority': 'surface'}, inplace=True)

df = df.merge(temp, how='left', left_on='old_ethnic', right_on='old_ethnic')

#%%
# Save the data
df.to_csv('data/zonal-stats.csv', index=False)
