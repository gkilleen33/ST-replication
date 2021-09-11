# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:59:32 2021

@author: Grady
"""

import ee
import math 
import pandas as pd
import os

ee.Initialize()

shapefile = ee.FeatureCollection('users/gradykilleen/slave-trade') 
aoi = shapefile.geometry().bounds()

# Rainfall 
gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V06').filter(ee.Filter.date('2014-01-01', '2018-12-31'))
gpm_scale = gpm.first().select('precipitation').projection().nominalScale().getInfo()

# Calculate total rainfall by year 
dates = [ee.Date('2014-01-01'), ee.Date('2015-01-01'), ee.Date('2016-01-01'), 
         ee.Date('2017-01-01'), ee.Date('2018-01-01')]
list_of_years = ee.List(dates)

def meanComposite(date, newlist):
	# Cast values
    date = ee.Date(date)
    newlist = ee.List(newlist)
    
    # Filter collection between date and the next day
    filtered_year = gpm.filterDate(date, date.advance(1,'year').advance(-1, 'day'))
    image = filtered_year.mean().select('precipitation').clip(aoi).set({'Date': date}).rename('rainfall')

    # Add the mosaic to a list only if the collection has images
    return ee.List(ee.Algorithms.If(filtered_year.size(), newlist.add(image), newlist))

rain_by_year = ee.ImageCollection(ee.List(list_of_years.iterate(meanComposite, ee.List([]))))

def zonalStats(image):
    date = image.get("Date")
    toReturn = image.reduceRegions(reducer=ee.Reducer.mean(), collection=shapefile, 
                                   scale=gpm_scale, tileScale=4)
    return toReturn.set('Date', date)

zs_rainfall = rain_by_year.map(zonalStats)

# Topographical diversity 
srtmTopographicDiversity = ee.Image('CSP/ERGo/1_0/Global/SRTM_topoDiversity').select('constant').clip(aoi)
maxPixels = math.ceil((3000/(srtmTopographicDiversity.projection().nominalScale().getInfo()))**2)
srtmTG_rescaled = srtmTopographicDiversity.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=maxPixels)
zs_top_div = srtmTG_rescaled.reduceRegions(reducer=ee.Reducer.mean(), collection=shapefile,
                                                    scale=3000, tileScale=4)

# Temperature 
temp = ee.Image("OpenLandMap/CLM/CLM_LST_MOD11A2-DAY_M/v01").clip(aoi)
average_temp = temp.reduce(ee.Reducer.mean())

zs_temp = average_temp.reduceRegions(reducer=ee.Reducer.mean(), collection=shapefile, 
                                     scale=average_temp.projection().nominalScale().getInfo(), tileScale=8)

# Remove geometry from each zonal stats and rename values
def processFeature(feature):
    return feature.setGeometry(None)

def removeGeometry(featureCollection):
    fc = ee.FeatureCollection(featureCollection)  # Cast
    fc_no_geometry = fc.map(processFeature)
    toReturn = fc_no_geometry.map(lambda x: x.set({'Date': fc.get('Date')}))
    return toReturn

zs_rain_ng = zs_rainfall.map(removeGeometry).flatten()

def renameRain(feature):
    return ee.Feature(None, {'old_ethnic': feature.get('old_ethnic'), 'Date': feature.get('Date'), 'rainfall': feature.get('mean')})

rain_final = zs_rain_ng.map(renameRain)

# Export rainfall separately since there are multiple years of data 
task = ee.batch.Export.table.toDrive(collection=rain_final, description='average-rainfall', 
                        fileFormat='CSV', fileNamePrefix='average-rainfall',
                        folder='earth-engine',
                        selectors=['old_ethnic', 'Date', 'rainfall'])

task.start()

def renameTD(feature):
    return ee.Feature(None, {'old_ethnic': feature.get('old_ethnic'), 'top_div': feature.get('mean')})

td_final = zs_top_div.map(renameTD).getInfo()['features']


def renameTemp(feature):
    return ee.Feature(None, {'old_ethnic': feature.get('old_ethnic'), 'av_temp': feature.get('mean')})

temp_final = zs_temp.map(renameTemp).getInfo()['features']

df = pd.DataFrame(columns=['old_ethnic', 'topological_diversity', 'average_temp'])

i = 0
for x in td_final:
    if 'top_div' in x['properties'].keys():
        df.loc[i] = (x['properties']['old_ethnic'], x['properties']['top_div'], None)
        i += 1

for x in temp_final:
    if 'av_temp' in x['properties'].keys():
        temp = x['properties']['av_temp']*.02*(9/5)-459.67  # .02 scaling factor, other calculation from K to F
        df.loc[(df['old_ethnic'] == x['properties']['old_ethnic']), 'average_temp'] = temp
    
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')

df.to_csv('data/top_div-temp.csv', index=False)
