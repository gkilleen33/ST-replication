# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:21:30 2021

@author: Grady
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os 
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extractOne

#%% 
# Set the working directory to the parent folder of this script 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')

#%%
# Import shapefile data 
old = gpd.read_file('data/murdock_shapefile/borders_tribes.shp')
new = gpd.read_file('data/etnicity_felix/etnicity_felix.shp')

old.set_index('NAME', drop=False, inplace=True)

#%%
# Merge the geometries of ethnicities with the same name
new['ethnicity_id'] = new['ETHNICITY'].astype('category')
new_dissolved = new.dissolve(by='ethnicity_id')

#%% Count how many ethnicities from Murdock have a match in the new data
# Commented out for speed: results from last run Matched: 736 (0.881437125748503), Not matched 99 (0.118562874251497), Perfect matches 423 (0.5065868263473053)
"""
list_murdock = list(set(old.NAME))
list_murdock = [x for x in list_murdock if 'UNINHABITED' not in x]

list_new = list(set(new_dissolved.ETHNICITY))
list_variant = list(set(new_dissolved.VARIANT))

match_number = 0
no_match_number = 0
perfect_match_number = 0

for x in list_murdock:
    if extractOne(x, list_new)[1] > 80:
        match_number += 1
        if extractOne(x, list_new)[1] == 100:
            perfect_match_number += 1
    elif extractOne(x, list_variant)[1] > 80:
        match_number += 1
        if extractOne(x, list_variant)[1] == 100:
            perfect_match_number += 1
    else:
        no_match_number += 1

print('Matched: {} ({}), Not matched {} ({}), Perfect matches {} ({})'.format(match_number, 
                                                                              match_number/len(list_murdock),
                                                                              no_match_number,
                                                                              no_match_number/len(list_murdock),
                                                                              perfect_match_number,
                                                                              perfect_match_number/len(list_murdock)))
"""

#%% 
# Now for each historical ethnic group, record the most prevalent modern ethnic group in the geographic area 

new_dissolved.set_index('ETHNICITY', inplace=True, drop=False)
joined = gpd.sjoin(old, new_dissolved, how='left', op='intersects')
joined['intersection_percent'] = 0

for i in list(set(joined.index)):
    if isinstance(joined.loc[i], gpd.GeoDataFrame):
        geom = joined.loc[i].geometry[0]  # Old geometry
        rows = joined.loc[i]
        for i, row in rows.iterrows():  # Loop over every matched new geom and calculate highest overlap
            if not pd.isnull(row['ETHNICITY']):
                percent_overlap = (geom.intersection(new_dissolved.loc[row['ETHNICITY']].geometry).area/geom.area)*100
                joined.loc[(joined.index == i) & (joined['ETHNICITY'] == row['ETHNICITY']), 'intersection_percent'] = percent_overlap
    elif isinstance(joined.loc[i], gpd.GeoSeries) or isinstance(joined.loc[i], pd.Series):
        # Only 1 match, not multiple 
        geom = joined.loc[i].geometry  # Old geometry
        if not pd.isnull(joined.loc[i]['ETHNICITY']):
           percent_overlap = (geom.intersection(new_dissolved.loc[joined.loc[i]['ETHNICITY']].geometry).area/geom.area)*100
           joined.loc[i, 'intersection_percent'] = percent_overlap
           
    else:
        raise Exception('Error: not series or geodataframe')
 
joined.rename(columns={'NAME': 'old_ethnicity'}, inplace=True)

# Now generate an indicator for whether the ethnicities match based on text proximity 
joined['score'] = joined.apply(lambda x: fuzz.token_set_ratio(x.old_ethnicity, x.ETHNICITY), axis=1)
joined['variant_score'] = joined.apply(lambda x: fuzz.token_set_ratio(x.old_ethnicity, x.VARIANT), axis=1)

joined['persistent'] = np.where((joined['score'] > 80) | (joined['variant_score'] > 80), 1, 0)

joined_continuous = joined.copy(deep=True)  # Continuous persistence rather than binary

joined.sort_values(by=['old_ethnicity', 'intersection_percent', 'persistent'], axis=0, ascending=False, inplace=True)
joined.drop_duplicates(subset='old_ethnicity', keep='first', inplace=True)
joined.reset_index(inplace=True, drop=True, level=0)

joined_continuous.sort_values(by=['old_ethnicity', 'persistent', 'intersection_percent'], axis=0, ascending=False, inplace=True)
joined_continuous.drop_duplicates(subset='old_ethnicity', keep='first', inplace=True)
joined_continuous.reset_index(inplace=True, drop=True, level=0)
joined_continuous.loc[joined_continuous['persistent'] == 0, 'intersection_percent'] = 0
joined_continuous.rename(columns={'intersection_percent': 'persistence_continuous'}, inplace=True)
joined_continuous = joined_continuous[['old_ethnicity', 'persistence_continuous']]

# Now merge continuous persistence measure back to main data 
joined = joined.merge(joined_continuous, how='left', left_on='old_ethnicity', right_on='old_ethnicity')

print(joined['persistent'].value_counts())

#%% Now merge in slave trade data 
slave_trade_df = pd.read_stata('data/tribe_level_slave_exports_Atlantic_Indian.dta')

joined.drop(columns=['index_right'])

final_df = joined.merge(slave_trade_df, how='left', left_on='old_ethnicity', right_on='murdock_name')
final_df['total_enslaved'] = final_df['atlantic_all_years'] + final_df['indian_all_years']
final_df['slave_trade'] = np.where(final_df['total_enslaved'] > 0, 1, 0)

final_df.drop(columns=['index_right', 'TRIBE_CODE', 'LAT', 'LON', 'ID', 'symbol', 
                       'score', 'variant_score', 'land_area'], inplace=True)

print(pd.crosstab(final_df['persistent'], final_df['slave_trade']))

#%% Add in the modern day country within which the historical boundary (primarily) falls in
lsib = gpd.read_file('data/lsib-simplified.geojson')
lsib = lsib[lsib['WLD_RGN'] == 'Africa']

temp = gpd.sjoin(lsib, final_df, how='left', op='intersects')
temp = temp.merge(final_df[['old_ethnicity', 'geometry']], how='left', left_on='old_ethnicity', right_on='old_ethnicity')

def overlap(row):
    try:
        overlap = (row['geometry_x'].intersection(row['geometry_y']).area/row['geometry_y'].area)*100
    except:
        overlap = 0
    return overlap

temp['overlap'] = temp.apply(lambda x: overlap(x), axis=1)
temp.sort_values(by=['old_ethnicity', 'overlap'], axis=0, ascending=False, inplace=True)
temp.drop_duplicates(subset='old_ethnicity', keep='first', inplace=True)
temp['country'] = temp['COUNTRY_NA']
temp = temp[['country', 'old_ethnicity']]

final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_on='old_ethnicity')

#%% Indicator for West Africa
un = gpd.read_file('data/countries_shp/countries.shp')
un = un[un['CONTINENT'] == 'Africa']

temp = gpd.sjoin(un, final_df, how='left', op='intersects')
temp = temp.merge(final_df[['old_ethnicity', 'geometry']], how='left', left_on='old_ethnicity', right_on='old_ethnicity')

temp['overlap'] = temp.apply(lambda x: overlap(x), axis=1)
temp.sort_values(by=['old_ethnicity', 'overlap'], axis=0, ascending=False, inplace=True)
temp.drop_duplicates(subset='old_ethnicity', keep='first', inplace=True)
temp['region'] = temp['UNREG1']
temp['west_africa'] = 0
temp.loc[temp.region == 'Western Africa', 'west_africa'] = 1
temp = temp[['west_africa', 'old_ethnicity']]

final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_on='old_ethnicity')

#%% Add in area 
temp = final_df.copy(deep=True)
# Reproject to Albers Equal Area conic centered in Africa, with meter units
temp = temp.to_crs('+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs')
temp['area'] = temp.geometry.area/10**6  # Area in square kilometers
temp = temp[['old_ethnicity', 'area']]
final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_on='old_ethnicity')


#%% Add colonizer data 
colonizers = gpd.read_file('data/colonizers/Colonial_Africa.shp')
colonizers = colonizers.dissolve(by='Colonizer')
temp = gpd.sjoin(final_df, colonizers, how='left', op='intersects')
temp = temp[['old_ethnicity', 'index_right']]
temp.rename(columns={'index_right': 'colonizer'}, inplace=True)
temp = temp.replace(np.nan, 'Independent', regex=True)
temp = temp.groupby(['old_ethnicity']).agg(lambda x: tuple(x)).applymap(list).reset_index()
temp['num_colonizers'] = temp.apply(lambda x: len(x.colonizer), axis=1)
colonizers = pd.get_dummies(pd.DataFrame([*temp.colonizer], index=temp.index).stack()).any(level=0).astype(int)
temp = temp.merge(colonizers, how='left', left_index=True, right_index=True)
temp.drop(columns=['colonizer'], inplace=True)
final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_on='old_ethnicity')

#%% Similarly add in the number of known gold deposits 
gold = gpd.read_file('data/gold/mrds-0-Gold.shp')
temp = gpd.sjoin(final_df, gold, how='inner', op='intersects')
temp['gold_dep'] = 1
temp = temp[['old_ethnicity', 'gold_dep']]
temp = temp.groupby('old_ethnicity').sum()
final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_index=True)
final_df['gold_dep'].fillna(0, inplace=True)

#%% Add in diamond deposits
diamonds = gpd.read_file('data/diamonds/DIADATA.shp')
temp = gpd.sjoin(final_df, diamonds, how='inner', op='intersects')
temp['diamonds'] = 1
temp = temp[['old_ethnicity', 'diamonds']]
temp = temp.groupby('old_ethnicity').sum()
final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_index=True)
final_df['diamonds'].fillna(0, inplace=True)

#%% Add in petroleum data 
oil = gpd.read_file('data/petroleum/PETRO_Onshore_080907.shp')
temp = gpd.sjoin(final_df, oil, how='inner', op='intersects')
temp['oil'] = 1
temp = temp[['old_ethnicity', 'oil']]
temp = temp.groupby('old_ethnicity').max()
final_df = final_df.merge(temp, how='left', left_on='old_ethnicity', right_index=True)
final_df['oil'].fillna(0, inplace=True)

#%% Add in a count of the number of bordering ethnicities (measure of fragmentation)
final_df['neighbors'] = None
for i, row in final_df.iterrows():   

    # get 'not disjoint' countries
    neighbors = final_df[~final_df.geometry.disjoint(row.geometry)].old_ethnicity.tolist()

    # remove own name of the country from the list
    neighbors = [ name for name in neighbors if row.old_ethnicity != name ]

    # add names of neighbors as NEIGHBORS value
    final_df.at[i, 'neighbors'] = len(neighbors)
    
#%% Add in the lat and lon of the centroid of each polygon 
final_df['lat'] = final_df.geometry.centroid.y
final_df['lon'] = final_df.geometry.centroid.x

#%% Save the final data
final_df.to_file('data/final_shp/slave_trade.shp')

google_ee_df = final_df[['old_ethnicity', 'geometry']]
google_ee_df.to_file('data/final_shp/google_ee.shp')


