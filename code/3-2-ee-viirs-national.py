# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:10:09 2021

@author: Grady

Calculates annual nightlights values by African nation from 2014-2018 using VIIRS data from the Google Earth Engine
"""

import ee 
import math

ee.Initialize()

###############################################################
# ENTER USER INPUTS HERE  
###############################################################
# Note: the default shapefile contains national boundary data
shapefile = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('wld_rgn', 'Africa'))

# Start and end dates for image search (YYYY-MM-DD)
begin = ee.Date('2014-01-01') # E.g. '2019-08-01'
end = ee.Date('2018-12-31') # E.g. '2019-12-15'

# Resolution: one of global (e.g. cross country analysis), national (e.g. across US states), regional (e.g. across US counties), or local (native resolution of 15 arc seconds)
resolution = 'regional' 

# Statistic: 'mean' or 'sum'
statistic = 'sum'

# Minimum number of cloud free days in composite (>= 1)
# More implies higher quality but lower coverage, especially in tropical areas
min_passes = 5

# Export information (to Google Drive)
output_folder = 'earth-engine'  # Folder name to save outputs in Google drive. The folder should be created before running the script.
output_file = 'viirs-national' # Output file name  

##############################################################
# END USER INPUTS 
##############################################################

aoi = shapefile.geometry().bounds()

# Import nightlights data
lights = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filter(ee.Filter.date(begin, end)).filterBounds(aoi)

# Make sure that min_passes is at least 1 
if min_passes < 1:
  raise Exception("min_passes must be at least 1")
  
if statistic == 'mean':
    reducer = ee.Reducer.mean()
elif statistic == 'sum':
    reducer = ee.Reducer.sum()
else: 
    raise Exception("Variable 'statistic' must be set to either 'mean' or 'sum'")
    
# Determine scale for zonal stats from resolution choice, and resample the image if necessary 
if resolution == 'global':
  scale = 40000
  tileScale = 4
  maxPixels = math.ceil((scale/(lights.first().projection().nominalScale().getInfo()))**2)
  lights = lights.map(lambda x: x.reduceResolution(reducer=reducer, maxPixels=maxPixels))
elif resolution == 'national':
  scale = 16000
  tileScale = 4
  maxPixels = math.ceil((scale/(lights.first().projection().nominalScale().getInfo()))**2)
  lights = lights.map(lambda x: x.reduceResolution(reducer=reducer, maxPixels=maxPixels))
elif resolution == 'regional':
  scale = 3000
  tileScale = 8
  maxPixels = math.ceil((scale/(lights.first().projection().nominalScale().getInfo()))**2)
  lights = lights.map(lambda x: x.reduceResolution(reducer=reducer, maxPixels=maxPixels))
elif resolution == 'local':
  scale = lights.first().projection().nominalScale().getInfo()
  tileScale = 16
else:
  raise Exception('Variable "resolution" must be specified')

    
# Mask out pixels with fewer satellites passes than min_passes in the composite 
def maskImage(image):
  image = ee.Image(image)  # Cast 
  date = image.get('system:index')  # The composites are indexed by date
  nightlights = image.select('avg_rad')
  below_zero = nightlights.gte(0)
  nightlights_clipped = nightlights.updateMask(below_zero).unmask() # Replace values lower than 0 with 0
  passes = image.select('cf_cvg')
  mask = ee.Image(passes).gte(min_passes)
  return ee.Image(nightlights_clipped).updateMask(mask).set({'Date': date}).clip(aoi) 

masked_nightlights = lights.map(maskImage)


# Create a median composite by year
dates = [ee.Date('2014-01-01'), ee.Date('2015-01-01'), ee.Date('2016-01-01'), 
         ee.Date('2017-01-01'), ee.Date('2018-01-01')]
list_of_years = ee.List(dates)

def medianComposite(date, newlist):
	# Cast values
    date = ee.Date(date)
    newlist = ee.List(newlist)
    
    # Filter collection between date and the next day
    filtered_year = masked_nightlights.filterDate(date, date.advance(1,'year').advance(-1, 'day'))
    image = filtered_year.median().select('avg_rad').clip(aoi).set({'Date': date}).rename('nightlights')

    # Add the mosaic to a list only if the collection has images
    return ee.List(ee.Algorithms.If(filtered_year.size(), newlist.add(image), newlist))

nightlights_by_year = ee.ImageCollection(ee.List(list_of_years.iterate(medianComposite, ee.List([]))))

def zonalStats(image):
    date = image.get("Date")
    toReturn = image.reduceRegions(reducer=reducer, collection=shapefile, scale=scale, tileScale=tileScale)
    return toReturn.set('Date', date)

zonal_stats = nightlights_by_year.map(zonalStats)

#  Remove geometry from the zonal stats
def processFeature(feature):
    return feature.setGeometry(None)

def removeGeometry(featureCollection):
    fc = ee.FeatureCollection(featureCollection)  # Cast
    fc_no_geometry = fc.map(processFeature)
    toReturn = fc_no_geometry.map(lambda x: x.set({'Date': fc.get('Date')}))
    return toReturn

zonal_stats_no_geometry = zonal_stats.map(removeGeometry).flatten()

# Export the data to Google Drive
task = ee.batch.Export.table.toDrive(collection=zonal_stats_no_geometry, description=output_file, 
                        fileFormat='CSV', fileNamePrefix=output_file,
                        folder=output_folder)

task.start()    
