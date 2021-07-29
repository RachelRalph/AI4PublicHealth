#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import geopandas as gpd
import pandas as pd 
import numpy as np 
import math
import copy
from operator import attrgetter
import time 


# In[2]:


ROUTES = gpd.read_file("C:/Users/jocel/OneDrive/Documents/External hard drive files/Documents/Calgary 21/Seattle GSP Program/GIS data/CLIPPED DATA MZUZU/MZUZU_roads_pointdata_with_elevation.shp")
ROUTE_DATA = pd.DataFrame(ROUTES)
ROUTES.head(10)


# In[3]:


ROUTES["x"] = ROUTES.geometry.x
ROUTES["y"] = ROUTES.geometry.y
print(ROUTES) 


# In[4]:


def preprocessing(unprocessed_raw_data):
    processed_raw_data = unprocessed_raw_data.drop("fid", axis=1)
    #processed_raw_data = unprocessed_raw_data.drop("fid", axis=1)
    processed_raw_data = processed_raw_data.reset_index()
    processed_data = []
    for rows in range(len(processed_raw_data.index)):
        #route_time = processed_raw_data.iloc[rows,1]
        coordinates = list(processed_raw_data.iloc[rows, 22].coords) #maybe take out the start lat and long here if we combine the dataframes for the line and point data 
        start_longitude = round(coordinates[0][0], 5)
        start_latitude = round(coordinates[0][1], 5)
        elevation = processed_raw_data.iloc[rows,17]
        road_condition = processed_raw_data.iloc[rows,10]
        road_type = processed_raw_data.iloc[rows,9]
        processed_data.append((start_longitude, start_latitude, elevation, road_condition, road_type))
    processed_data = pd.DataFrame(processed_data)
    processed_data = processed_data.rename(columns ={0:"start_longitude", 1:"start_latitude", 2:"elevation", 3:"road_condition", 4:"road_type"})
    return processed_data
#print(ROUTE_DATA.iloc[200,13])    
clean_data = preprocessing(ROUTE_DATA)
print(clean_data)


# In[5]:


ROUTES_line = gpd.read_file("C:/Users/jocel/OneDrive/Documents/External hard drive files/Documents/Calgary 21/Seattle GSP Program/GIS data/CLIPPED DATA MZUZU/MZUZU_roads_lines_CORRECT.shp")
ROUTE_DATA_line = pd.DataFrame(ROUTES_line)
ROUTES_line.head(60)
#print(ROUTE_DATA_line['geometry'])


# In[14]:


#data = []
#for i in range(33):
  #linestring = df.iloc[i, 2]
  #time = df.iloc[i,1]
  #coordinates = list(df.iloc[i, 2].coords)
  #coorX1 = round(coordinates[0][0], 3)
  #coorY1 = round(coordinates[0][1], 3)

  #coorX2 = round(coordinates[-1][0], 3)
  #coorY2 = round(coordinates[-1][1], 3)
  #data.append([time, [coorX1, coorY1] , [coorX2, coorY2]])

#linestring = ROUTE_DATA_line.iloc[rows,11]

print(ROUTE_DATA_line)


# In[7]:


ROUTE_DATA_line.iloc[59, 11]


# In[8]:


#import fiona
#from shapely.geometry import shape
#with fiona.open("C:/Users/jocel/OneDrive/Documents/External hard drive files/Documents/Calgary 21/Seattle GSP Program/GIS data/CLIPPED DATA MZUZU/MZUZU_roads_lines_CORRECT.shp") as copy_shpfile:
    #for feature in copy_shpfile:
        #geom = feature['geometry']
        # geom in GeoJSON format -> {'type': 'MultiLineString', 'coordinates': (((0.0, 0.0), (1.0, 1.0)), ((-1.0, 0.0), (1.0, 0.0)))}
        #if geom['type'] == 'MultiLineString':
               # convert to shapely geometry
               #shapely_geom = shape(geom) # = MULTILINESTRING ((0 0, 1 1), (-1 0, 1 0))
               #for lines in shapely_geom:
                   #print(lines)


# In[15]:


def preprocessing_line(unprocessed_raw_data):
    

    processed_data_line = []
    for rows in range(len(unprocessed_raw_data.index)):
        coordinates_line = unprocessed_raw_data.iloc[rows, 11]
        string_type = (type(coordinates_line))
        
        if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
            coordinates_line = list(coordinates_line.coords)
            start_longitude_line = round(coordinates_line[0][0], 5)
            start_latitude_line = round(coordinates_line[0][1], 5)
            end_longitude_line = round(coordinates_line[-1][0], 5)
            end_latitude_line = round(coordinates_line[-1][1], 5)
            #elevation = processed_raw_data.iloc[rows,17]
            #road_condition = processed_raw_data.iloc[rows,10]
            #road_type = processed_raw_data.iloc[rows,9]
            processed_data_line.append((start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))
            
        elif str(string_type) != "<class 'shapely.geometry.linestring.LineString'>":
            print(string_type)
            #print(coordinates_line)
            
            for item in coordinates_line:
                #x_coords = [list(x.coords) for x in list(item)]
                #print(item)
                coordinates_line = item
                coordinates_line = list(coordinates_line.coords)
                start_longitude_line = round(coordinates_line[0][0], 5)
                start_latitude_line = round(coordinates_line[0][1], 5)
                end_longitude_line = round(coordinates_line[-1][0], 5)
                end_latitude_line = round(coordinates_line[-1][1], 5)
                #elevation = processed_raw_data.iloc[rows,17]
                #road_condition = processed_raw_data.iloc[rows,10]
                #road_type = processed_raw_data.iloc[rows,9]
                processed_data_line.append((start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))             
        #else: 
    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(columns ={0:"start_longitude", 1:"start_latitude", 2:"end_longitude_line", 3:"end_latitude_line"})
    return processed_data_line

clean_line_data = preprocessing_line(ROUTE_DATA_line)
print(clean_line_data)


# In[17]:


def preprocessing_line(unprocessed_raw_data):
    #processed_raw_data = unprocessed_raw_data.drop("fid", axis=1)
    #processed_raw_data = unprocessed_raw_data.drop("fid", axis=1)
    #processed_raw_data = processed_raw_data.reset_index()
    

    processed_data_line = []
    for rows in range(len(unprocessed_raw_data.index)):
        #route_time = processed_raw_data.iloc[rows,1]
        coordinates_line = unprocessed_raw_data.iloc[rows, 11]
        #print(rows, coordinates_line.length)
        string_type = (type(coordinates_line))
        if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
            coordinates_line = list(coordinates_line.coords)
            start_longitude_line = round(coordinates_line[0][0], 5)
            start_latitude_line = round(coordinates_line[0][1], 5)
            end_longitude_line = round(coordinates_line[-1][0], 5)
            end_latitude_line = round(coordinates_line[-1][1], 5)
            #elevation = processed_raw_data.iloc[rows,17]
            #road_condition = processed_raw_data.iloc[rows,10]
            #road_type = processed_raw_data.iloc[rows,9]
            processed_data_line.append((start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))
        elif str(string_type) != "<class 'shapely.geometry.linestring.LineString'>":
            print(string_type)
            
        #else: 
    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(columns ={0:"start_longitude", 1:"start_latitude", 2:"end_longitude_line", 3:"end_latitude_line"})
    return processed_data_line

clean_line_data = preprocessing_line(ROUTE_DATA_line)
print(clean_line_data)

from shapely.geometry import Point
clean_line_data['geometry'] = clean_line_data.apply(lambda x: Point((float(x.start_longitude), float(x.start_latitude))), axis=1)
import geopandas
clean_line_data = geopandas.GeoDataFrame(clean_line_data, geometry='geometry')
clean_line_data.to_file('MyGeometries.shp', driver='ESRI Shapefile')

