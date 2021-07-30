import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from shapely.geometry import Point
import time

# Get the start time.
START_TIME = time.time()

# Get script and dataset file paths.
SCRIPT_PATH = os.path.dirname(__file__)

# Read the node intersection geojson file via geopandas and store as a pandas dataframe.
NODE_INTERSECTION_PATH = os.path.join(SCRIPT_PATH, "Datasets/Clipped_Mzuzu_Road_Intersections.geojson")
NODE_INTERSECTION_DATA = gpd.read_file(NODE_INTERSECTION_PATH)
NODE_INTERSECTION_DATA = pd.DataFrame(NODE_INTERSECTION_DATA)

# Read the road points w/ elevation .shp file via geopandas and store as a pandas dataframe.
ROAD_POINT_WITH_ELEVATION_PATH = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_pointdata_with_elevation.shp")
ROAD_POINT_WITH_ELEVATION_DATA = gpd.read_file(ROAD_POINT_WITH_ELEVATION_PATH)
ROAD_POINT_WITH_ELEVATION_DATA = pd.DataFrame(ROAD_POINT_WITH_ELEVATION_DATA)

# Read the road line data .shp file via geopandas and store as a pandas dataframe.
ROAD_LINE_PATH = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_lines_CORRECT.shp")
ROAD_LINE_DATA = gpd.read_file(ROAD_LINE_PATH)
ROAD_LINE_DATA = pd.DataFrame(ROAD_LINE_DATA)
INCLUDE_MULTILINE = True


def intersection_processing(intersection_df):
    def intersection_processing(intersection_df):
        """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
            version of the original data from the .shp file. """

        # Create a secondary pandas data frame that contains the index of nodes, start/end longitude and latitude,
        # elevation, road condition, and road type.
        processed_data = []

        # TODO: Maybe there's a more efficient way to do this than to loop through the entire unprocessed data set
        for rows in range(len(intersection_df.index)):
            coordinates_line = intersection_df.iloc[rows, 2]
            # TODO: Check what the team meant by this comment
            # Maybe take out the start lat and long here if we combine the dataframes for the line and point data
            for item in coordinates_line:
                coordinates_line = item

                coordinates_line = list(coordinates_line.coords)

                start_longitude = coordinates_line[0][0]
                start_latitude = coordinates_line[0][1]

                processed_data.append((start_longitude, start_latitude))

        processed_data = pd.DataFrame(processed_data)
        processed_data = processed_data.rename(
            columns={0: "Longitude", 1: "Latitude"})

        return processed_data


def road_elevation_processing(road_elevation_df, intersection_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    # Create a secondary pandas data frame that contains the index of nodes, start/end longitude and latitude,
    # elevation, road condition, and road type.
    processed_data = []

    intersection_longitude_list = intersection_df["Longitude"].values
    intersection_latitude_list = intersection_df["Latitude"].values

    # TODO: Maybe there's a more efficient way to do this than to loop through the entire unprocessed data set
    for rows in range(len(road_elevation_df.index)):
        # TODO: Check what the team meant by this comment
        # Maybe take out the start lat and long here if we combine the dataframes for the line and point data
        coordinates = list(road_elevation_df.iloc[rows, 22].coords)
        start_longitude = coordinates[0][0]
        start_latitude = coordinates[0][1]

        elevation = road_elevation_df.iloc[rows, 17]
        road_condition = road_elevation_df.iloc[rows, 10]
        road_type = road_elevation_df.iloc[rows, 9]

        if start_longitude in intersection_longitude_list:
            if start_latitude in intersection_latitude_list:
                processed_data.append((start_longitude, start_latitude, elevation, road_condition, road_type, True))
        else:
            processed_data.append((start_longitude, start_latitude, elevation, road_condition, road_type, False))

    processed_data = pd.DataFrame(processed_data)

    processed_data = processed_data.rename(
        columns={0: "Longitude", 1: "Latitude", 2: "Elevation", 3: "Road Condition", 4: "Road Type",
                 5: "Intersection Node"})

    return processed_data


def road_line_processing(road_line_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    processed_data_line = []

    for rows in range(len(road_line_df.index)):
        coordinates_line = road_line_df.iloc[rows, 11]
        string_type = (type(coordinates_line))

        if INCLUDE_MULTILINE:
            if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
                coordinates_line = list(coordinates_line.coords)

                start_longitude_line = coordinates_line[0][0]
                start_latitude_line = coordinates_line[0][1]
                end_longitude_line = coordinates_line[-1][0]
                end_latitude_line = coordinates_line[-1][1]

                processed_data_line.append(
                    (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))

            elif str(string_type) != "<class 'shapely.geometry.linestring.MultiLineString'>":
                for item in coordinates_line:
                    coordinates_line = item

                    coordinates_line = list(coordinates_line.coords)

                    start_longitude_line = coordinates_line[0][0]
                    start_latitude_line = coordinates_line[0][1]
                    end_longitude_line = coordinates_line[-1][0]
                    end_latitude_line = coordinates_line[-1][1]

                    processed_data_line.append(
                        (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))

            else:
                print("There is a unique string type that is neither LineString or MultiString:")
                print("    ", string_type)

        else:
            if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
                coordinates_line = list(coordinates_line.coords)

                start_longitude_line = coordinates_line[0][0]
                start_latitude_line = coordinates_line[0][1]
                end_longitude_line = coordinates_line[-1][0]
                end_latitude_line = coordinates_line[-1][1]

                processed_data_line.append(
                    (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line))

            else:
                continue

    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(
        columns={0: "Start Longitude", 1: "Start Latitude", 2: "End Longitude", 3: "End Latitude"})

    return processed_data_line


def confirmation_intersection_points(intersection_df, road_elevation_df):
    intersection_confirm = []

    # Get the rows from road_elevation_df that contain 'True' for the node intersection column.
    for row in range(len(road_elevation_df.index)):
        if road_elevation_df.iloc[row][5]:
            lat = road_elevation_df.iloc[row, 0]
            long = road_elevation_df.iloc[row, 1]
            intersection_confirm.append([lat, long])
        else:
            continue

    # Make the list into a dataframe, give it column names, reset the index, drop the extra index column,
    # and sort the values.
    intersection_confirm = pd.DataFrame(intersection_confirm)
    intersection_confirm = intersection_confirm.rename(
        columns={0: "Longitude", 1: "Latitude"})
    intersection_confirm = intersection_confirm.reset_index()
    intersection_confirm = intersection_confirm.drop("index", axis=1)
    intersection_confirm.sort_values(by=["Longitude"], ascending=False)

    # Create another list to contain the rows that aren't present in both lists.
    not_equal = []

    # Get the difference between the number of rows in intersection_row and the intersection_confirm.
    compare_val = len(intersection_df.index) - len(intersection_confirm.index)

    # Make two sublist containing just the longitude and latitude from the intersection_confirmation datafrane.
    intersection_confirm_longitude_list = intersection_confirm["Longitude"].values
    intersection_confirm_latitude_list = intersection_confirm["Latitude"].values

    for row in range(len(intersection_df.index) - compare_val):
        if intersection_df.iloc[row, 0] in intersection_confirm_longitude_list:
            if intersection_df.iloc[row, 1] in intersection_confirm_latitude_list:
                continue
            else:
                not_equal.append(intersection_df.iloc[row])
        else:
            not_equal.append(intersection_df.iloc[row])

    not_equal = pd.DataFrame(not_equal)
    not_equal = not_equal.rename(
        columns={0: "Longitude", 1: "Latitude"})
    not_equal = not_equal.reset_index()
    not_equal = not_equal.drop("index", axis=1)
    not_equal.sort_values(by=["Longitude"], ascending=False)

    return not_equal, intersection_confirm


def main():
    intersection_nodes = intersection_processing(NODE_INTERSECTION_DATA)
    intersection_nodes.sort_values(by=["Longitude"], ascending=False)

    road_elevation_nodes = road_elevation_processing(ROAD_POINT_WITH_ELEVATION_DATA, intersection_nodes)
    road_line_nodes = road_line_processing(ROAD_LINE_DATA)

    not_equal_nodes, intersection_confirm_nodes = confirmation_intersection_points(intersection_nodes,
                                                                                   road_elevation_nodes)

    print("INTERSECTION NODES:")
    print(intersection_nodes)

    print("\nROAD ELEVATION NODES:")
    print(road_elevation_nodes)

    print("\nROAD LINE NODES:")
    print(road_line_nodes)

    print("\nINTERSECTION CONFIRMATION NODES:")
    print(intersection_confirm_nodes)

    print("\nNOT EQUAL NODES:")
    print(not_equal_nodes)

    print()
    print("\n------------------------")
    print("Minutes since execution:", (time.time() - START_TIME) / 60)


if __name__ == "__main__":
    main()
