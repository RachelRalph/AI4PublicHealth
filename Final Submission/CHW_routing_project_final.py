# Imports
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from math import sin, cos, radians, pi
from pyproj import Geod
import copy
import sys
import random
from random import randint
from operator import attrgetter
import matplotlib.pyplot as plt
import networkx as nx
import time
import pickle

# Get the start time.
START_TIME = time.time()

# General program parameters.
LOAD_DATA = True
VISUALIZATION = True
ALGORITHM = 1


if not LOAD_DATA:
    # Get script and dataset file paths.
    SCRIPT_PATH = os.path.dirname(__file__)

    # Read the building point data w/ elevation .shp file via geopandas and store as a pandas dataframe.
    BUILDING_FILE_WITH_ELEVATION = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_buildings_with_elevation.shp")
    BUILDING_FILE_WITH_ELEVATION = gpd.read_file(BUILDING_FILE_WITH_ELEVATION)
    BUILDING_FILE_WITH_ELEVATION = pd.DataFrame(BUILDING_FILE_WITH_ELEVATION)

    # Read the road line data .shp file via geopandas and store as a pandas dataframe.
    ROAD_LINE_DATA = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_lines_CORRECT.shp")
    ROAD_LINE_DATA = gpd.read_file(ROAD_LINE_DATA)
    ROAD_LINE_DATA = pd.DataFrame(ROAD_LINE_DATA)

    # Read the road points with elevation .shp file via geopandas and store as a pandas dataframe.
    ROAD_POINT_WITH_ELEVATION_PATH = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_pointdata_with_elevation.shp")
    ROAD_POINT_WITH_ELEVATION_DATA = gpd.read_file(ROAD_POINT_WITH_ELEVATION_PATH)
    ROAD_POINT_WITH_ELEVATION_DATA = pd.DataFrame(ROAD_POINT_WITH_ELEVATION_DATA)

# Data processing
"""
Note on data preprocessing: 

The road node data in OpenStreetMaps is unusable because the bearings and distances of each node are incredibly 
unreliable. The line data is much better, however for our algorithm to work, we must convert the line data into node 
data. Below is our code to process each road into a list of coordinates, and from there we use the roads to deduce 
each nodes connections (the nodes next to them on the roads), and which nodes are intersections. 

"""


def road_line_processing(road_line_df):
    """Process road line data.

    Clean the .shp file that contains the route data. Create a second Pandas DataFrame to store a processed version
    of the original data from the .shp file. Returns the second Pandas DataFrame.

    :param pd.DataFrame road_line_df: Pandas DataFrame that contains the road line data.
    :returns: processed_data_line: Pandas DataFrame containing the start, intermediate, and end coordinates contained
                                   within the road line data for each road.

    """

    processed_data_line = []

    # Stores coordinates from geometry.
    for index, rows in road_line_df.iterrows():
        coordinates_line = road_line_df.iloc[index, 11]
        string_type = (type(coordinates_line))

        # Exclude MultiString elements.
        if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
            coordinates_line = list(coordinates_line.coords)

            start_longitude_line = round(coordinates_line[0][0], 6)
            start_latitude_line = round(coordinates_line[0][1], 6)
            end_longitude_line = round(coordinates_line[-1][0], 6)
            end_latitude_line = round(coordinates_line[-1][1], 6)

            all_coord = []

            # Appends all coordinates from the LineString geometry into a list.
            for coordinate in coordinates_line:
                all_coord.append(coordinate)

            processed_data_line.append(
                (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line, all_coord))

    # Convert to a DataFrame with the start coordinates of every road, and a coordinate list containing all coordinates.
    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(
        columns={0: "Start Longitude", 1: "Start Latitude", 2: "End Longitude", 3: "End Latitude",
                 4: "Coordinates List"})

    # Print the time taken to run the up to the current function.
    print("Road Line Processing Time:", (time.time() - START_TIME) / 60)

    # Return processed_data_line.
    return processed_data_line


def line_node_dataframe(road_line_df):
    """Nodes from road line data.

    This function creates a DataFrame where each set of coordinates has it's own row, with the coordinates that
    it's connecting to (or next to on the road) .

    :param pd.DataFrame road_line_df: Pandas DataFrame containing the start, intermediate, and end coordinates contained
                                      within the road line data for each road.
    :returns: node_data: Pandas DataFrame containing parent node coordinate and the coordinates of the sub-nodes
                         connecting to the parent node.

    """

    node_information = []

    # Iterate through all the rows in the road_line_df
    for index, row in road_line_df.iterrows():

        # Iterate through the number of connections for each row
        for i in range(len(row["Coordinates List"])):
            connections = []
            coordinate_pair = row["Coordinates List"][i]

            if i != 0:
                connections.append(row["Coordinates List"][i - 1])

            if i != len(row["Coordinates List"]) - 1:
                connections.append(row["Coordinates List"][i + 1])

            node_information.append([coordinate_pair, connections])

    node_data = pd.DataFrame(node_information)

    # Print the time taken to run the up to the current function.
    print("Node DataFrame Processing Time:", (time.time() - START_TIME) / 60)

    # Return node_data.
    return node_data


def road_elevation_processing(road_elevation_df):
    """Dictionary of nodes from road elevation point data.

    This function creates a dictionary where the longitude/latitude pairs are the keys, and the values are the
    elevation.

    :param pd.DataFrame road_elevation_df: Pandas DataFrame containing various pieces of information for each road node.
    :returns: elevation_dict: Dictionary containing the node longitude/latitude as the keys, and elevation as the value.

    """

    # Create a dictionary that contains latitude/longitude coordinates and the corresponding elevation.
    elevation_dict = {}

    # Iterate through each row of road_elevation_df
    for rows in range(len(road_elevation_df.index)):

        # Get the coordinates.
        coordinates = list(road_elevation_df.iloc[rows, 22].coords)
        start_latitude = coordinates[0][1]
        start_longitude = coordinates[0][0]

        # Get the elevation.
        elevation = road_elevation_df.iloc[rows, 17]

        elevation_dict[(start_longitude, start_latitude)] = elevation

    # Print the time taken to run the up to the current function.
    print("Elevation Dictionary Processing Time:", (time.time() - START_TIME) / 60)

    # Return the elevation_dict.
    return elevation_dict


def node_dictionary(node_df):
    """Dictionary and list from road line data nodes.

    This function converts the node_data into a dictionary where the longitude/latitude pairs are the keys. This
    function also logs all nodes it processes more that once in the multiple index pairs list. Because a node was
    processed more than once, it must include more than one row, ergo, multiple_index_pairs flags all the nodes that
    are intersections.

    :param pd.DataFrame node_df: Pandas DataFrame containing parent node coordinate and the coordinates of the sub-nodes
                                 connecting to the parent node.
    :returns: long_lat_pairs, multiple_index_pairs: Dictionary containing the longitude/latitude tuple as the key, and
                                                    the index as the value. List containing all of the
                                                    longitude/latitude coordinate tuples.

    """

    long_lat_pairs = {}

    multiple_index_pairs = []

    # Iterate through the rows in node_df.
    for index, row in node_df.iterrows():

        # Access the row at index zero and retrieve the longitude/latitude tuple.
        long_lat_coord = row[0]

        if long_lat_coord in long_lat_pairs:
            long_lat_pairs[long_lat_coord].append(index)

            if long_lat_coord not in multiple_index_pairs:
                multiple_index_pairs.append(long_lat_coord)

        else:
            long_lat_pairs[long_lat_coord] = [index]

    # Print the time taken to run the up to the current function.
    print("Node Dictionary Processing Time:", (time.time() - START_TIME) / 60)  # 0.3

    # Return the long_lat_pairs and multiple_index_pairs.
    return long_lat_pairs, multiple_index_pairs


def intersection_node_check(node_df, node_dict, index_pairs):
    """Check for intersection nodes.

    Sets an IsIntersection value in the dataframe for all intersections. Furthermore, a third value is set. This value
    is set for nodes that where processed more than once; and have an equivalent node flagged as an intersection in the
    dataframe. this happens so that we know which nodes we don't have to process more than once.

    :params: pd.DataFrame node_df: Pandas DataFrame containing parent node coordinate and the coordinates of the
                                   sub-nodes connecting to the parent node.
    :params: dict node_dict: Dictionary containing the longitude/latitude tuple as the key, and the index as the value
                             from the node data from the road line dataframe.
    :params: list index_pairs: List containing all of the longitude/latitude coordinate tuples.
    :returns: node_df: Pandas DataFrame containing parent node coordinate and the coordinates of the sub-nodes
                       connecting to the parent node. Also contains a boolean value if the node is an intersection.

    """

    # Create two more columns. Set the first to be True throughout and the second to be False.
    node_df[2] = True
    node_df[3] = False

    min_length = math.inf

    for pair in index_pairs:
        index_list = node_dict[pair]

        if len(index_list) < min_length:
            min_length = len(index_list)

        first_index = index_list[0]

        for second_index in index_list[1:]:
            all_connections = list(set(node_df.iat[first_index, 1] + node_df.iat[second_index, 1]))
            node_df.iat[first_index, 1] = all_connections
            node_df.iat[second_index, 2] = False
            node_dict[pair] = [first_index]

    for index, row in node_df.iterrows():
        if len(row[1]) != 2:
            node_df.iat[index, 3] = True

    # This loop combines the connections for all intersections, so all nodes in an intersection share the same
    # connections.
    for index, row in node_df.iterrows():
        if row[2]:
            if row[3]:
                continue
            else:
                if len(node_df.iat[index, 1]) >= 2:
                    connection1 = node_df.iat[index, 1][0]
                    connection2 = node_df.iat[index, 1][1]
                    index1 = node_dict[connection1][0]
                    index2 = node_dict[connection2][0]
                    node_df.at[index1, 1].remove(node_df.iat[index, 0])
                    node_df.at[index2, 1].remove(node_df.iat[index, 0])
                    # Converting between a list and a set ensures that only unique values with be retained.
                    node_df.at[index1, 1] = list(set(node_df.iat[index1, 1] + [node_df.iat[index2, 0]]))
                    node_df.at[index2, 1] = list(set([node_df.iat[index1, 0]] + node_df.iat[index2, 1]))

    index_list = []

    for index, row in node_df.iterrows():
        if not row[2]:
            index_list.append(index)

    for index in index_list:
        node_df = node_df.drop(index, axis=0)

    # Print the time taken to run the up to the current function.
    print("Node Intersection Processing Time:", (time.time() - START_TIME) / 60)  # 2.6

    # Return the node_df.
    return node_df


def intersection_node_dictionary(node_df):
    """Intersection node dictionary.

    This function creates a dictionary of intersection nodes, for the sake of easy look up.

    :params: pd.DataFrame node_df: Pandas DataFrame containing the start, intermediate, and end coordinates contained
                                   within the road line data for each road. Also contains a boolean value if the node is
                                   an intersection.
    :returns: intersection_node_dict: Dictionary containing the longitude/latitude as the keys and the values as the
                                      respective index number for all the intersection nodes.

    """

    intersection_node_dict = {}
    node_df = node_df.rename(
        columns={0: "Long/Lat Coordinates", 1: "Connections", 3: "Is Intersection", })

    for index, row in node_df.iterrows():
        if row["Is Intersection"]:
            row_node = Node(None, None, row["Long/Lat Coordinates"], row["Connections"], None)

            if LOAD_DATA:
                intersection_node_dict[row["Long/Lat Coordinates"]] = row_node

            else:
                intersection_node_dict[row["Long/Lat Coordinates"]] = row_node

    print("Intersection Node Dictionary Processing Time:", (time.time() - START_TIME) / 60)

    # Return the intersection_node_dict.
    return intersection_node_dict


def get_houses(number_of_houses):
    """Random house selection list.

    This function randomly selects a user selected number of houses to be utilized in the network generation.

    :params: int number_of_houses: The number of houses that you would like to create a route for.
    :returns: house_ids: A list of random houses that exist in Mzuzu.

    """

    house_ids = []

    # Seed can be changed to get different random numbers.
    random.seed(42)

    # Choose a random number, which corresponds to a house in the DataFrame of homes, and append to a list.
    for _ in range(number_of_houses):
        house_ids.append(random.randint(0, len(BUILDING_FILE_WITH_ELEVATION)))

    # Return the house_ids.
    return house_ids


# Get nearest intersection for houses, to use as a benchmark for algorithms below.
def nearest_intersection_to_house(houses, node_data, intersection_dict):
    """Nearest intersection for houses.

    Get nearest intersection for houses, to use as a benchmark for algorithms below.

    :params: list houses: A list of random houses that exist in Mzuzu.
    :params: pd.DataFrame node_data: Pandas DataFrame containing the start, intermediate, and end coordinates contained
                                     within the road line data for each road. Also contains a boolean value if the node
                                     is an intersection.
    :params: dict intersection_dict: Dictionary containing the longitude/latitude as the keys and the values as the
                                     respective index number for all the intersection nodes.
    :returns: nearest_intersections: List of the nearest intersection to the homes in the list of houses with their
                                     associated coordinates and cost of movement.

    """

    nearest_intersections = []
    node_data = node_data.rename(
        columns={0: "Long/Lat Coordinates", 1: "Connections", 3: "Is Intersection", })

    for house_id in houses:
        coordinates = list(BUILDING_FILE_WITH_ELEVATION.loc[house_id, "geometry"].coords)
        # Get lat/long coordinates for the house.
        long = coordinates[0][0]
        lat = coordinates[0][1]

        elevation = BUILDING_FILE_WITH_ELEVATION.at[house_id, "SAMPLE_1"]
        near_intersection = None
        near_intersection_dist = math.inf

        for index, row in node_data.iterrows():
            if row["Is Intersection"]:
                if LOAD_DATA:
                    node_long = row["Long/Lat Coordinates"][0]
                    node_lat = row["Long/Lat Coordinates"][1]

                else:
                    node_long = row["Long/Lat Coordinates"][0]
                    node_lat = row["Long/Lat Coordinates"][1]

                huer = heuristic(long, lat, node_long, node_lat, False)
                if near_intersection_dist > huer:
                    near_intersection_dist = huer
                    near_intersection = intersection_dict[(node_long, node_lat)]
        nearest_intersections.append(near_intersection)

    # Return nearest_intersections.
    return nearest_intersections


# This function finds the three nearest houses to perform an A* search on, so that we can find the quickest path
# between house A and house B.
def find_nearby_houses(nearest_intersections, intersection_dict, elevation_dict):
    """Dictionary of nearby houses.

    This function finds the three nearest houses to perform an A* search on, so that we can find the quickest path
    between house A and house B.

    :params: list nearest_intersections: List of the nearest intersection to the homes in the list of houses with their
                                         associated coordinates and cost of movement.
    :params: dict intersection_dict: Dictionary containing the longitude/latitude as the keys and the values as the
                                     respective index number for all the intersection nodes.
    :params: elevation_dict: Dictionary containing the node longitude/latitude as the keys, and elevation as the value.
    :returns: graph: Dictionary containing the three nearest houses to an intersection, which is the intersection
                     nearest to each of the randomly selected homes.

    """

    graph = {}

    for intersection in nearest_intersections:
        houses_completed = []
        houses_visited = []

        for _ in range(3):
            smallest_dist = math.inf
            closest_house = None

            for house_to_go in nearest_intersections:
                if house_to_go not in houses_completed and house_to_go != intersection:
                    dist = heuristic(house_to_go.long_lat[0], house_to_go.long_lat[1], intersection.long_lat[0],
                                     intersection.long_lat[1], True, elevation_dict)
                    if dist < smallest_dist:
                        smallest_dist = dist
                        closest_house = house_to_go
            cost = node_to_node_search(closest_house, intersection, intersection_dict, elevation_dict)
            houses_completed.append(closest_house)
            houses_visited.append((closest_house, cost))
            graph[intersection] = houses_visited

    return graph


# General classes
# Node class to use for traveling salesman problem.
class Node:

    def __init__(self, road_condition, elevation, long_lat, connections, node_id):
        self.road_condition = road_condition
        self.elevation = elevation
        self.connections = connections
        self.long_lat = long_lat
        self.f = None
        self.g = None
        self.h = None
        self.id = node_id

    # Convert function works to change connections from a list to a dictionary, depending on the algorithm. (A* uses
    # list, branch and bound uses a dictionary.)
    def convert_connections_for_graph(self):
        self.connections = {}


# Graph class.
class Graph:
    # One attribute, a list of all the nodes contained in the graph.
    def __init__(self, nodes_for_graph):
        self.nodes = nodes_for_graph

    # Function to convert the graph into an adjacency matrix. Where there are no connections, the value of the matrix
    # is set to infinity.
    def convert_to_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        matrix = np.where(matrix == 0, math.inf, matrix)
        for node in self.nodes:
            for connections in node.connections:
                matrix[node.id][connections.id] = node.connections[connections]
        return matrix

    # Returns size of graph.
    def size(self):
        return len(self.nodes)


# Priority queue class.
class PriorityQueue:
    # Priority queue is stored in a dictionary, in the format {node: [priority, matrix]}
    def __init__(self):
        self.queue = {}

    # Returns if priority queue is empty.
    def isEmpty(self):
        return len(self.queue) == 0

    # Push an item to the priority queue with a matrix
    def push_with_matrix(self, node, priority, matrix):
        self.queue[node] = [priority, matrix]

    # Push without a matrix
    def push_wo_matrix(self, node, priority):
        self.queue[node] = priority

    # Check to see if a node is already in the Queue.
    def inQueue(self, node):
        for node in self.queue:
            if node.id == node.id:
                return True
        return False

    # Pop a node and get the matrix.
    def pop_with_matrix(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node][0]
        matrix = self.queue[node][1]

        del self.queue[node]
        return node, cost, matrix

    # Peek and see the next node, without deleting the node from the priority queue.
    def peek(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        return node, cost

    # Pop without returning the matrix.
    def pop_wo_matrix(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        del self.queue[node]
        return node, cost


# Solid state node class, these are primarily used to keep track of what we have tried using the branch and bound
# method.
class SolidStateNode:

    def __init__(self, parent_node, node_id, node, level, reduced_matrix, cost):
        # Each SolidState node has a graph node, id, matrix, parent node, level and cost associated with them.
        self.node = node
        self.node_id = node_id
        self.reduced_matrix = reduced_matrix
        self.parent_node = parent_node
        self.level = level
        self.cost = cost
        self.children = []
        if parent_node is not None:
            parent_node.add_children(self)
            self.closedList = parent_node.closedList + [parent_node.node]

        else:
            self.closedList = []

    # Method to add a child.
    def add_children(self, child):
        self.children.append(child)


class SolidStateTree:
    def __init__(self, root):
        self.root = root


# General functions
def calculate_cost(matrix):
    """Calculate the matrix reduction cost.

    Calculates the reduction cost of the matrix. Returns the adjacency matrix (reduced matrix) and the associated
    cost.

    :params: list matrix: 2D list consisting of the edge weights between V*V nodes.
    :returns: cost, matrix: The row and column reduction cost, as well as the resulting reduced matrix, respectively.

    """

    cost = 0

    def row_reduction():
        """Reduce the rows. Each row is reduced based on the lowest number in itself. Returns the row reduced matrix."""

        # Get minimum values of all the rows. Note that this includes the zeros in the rows.
        min_values = np.min(matrix, axis=1)

        # Iterate through all of the rows and columns and subtract the row-respective min_value from all of the
        # numbers in the respective row. The subtraction throughout the matrix rows yields a row reduced matrix.
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[row]

        # Return matrix and min_values.
        return matrix, min_values

    def col_reduction():
        """Reduce the columns. Each column is reduced based on the lowest number in itself. Returns the column
        reduced matrix. """

        # Get minimum values of all the columns. Note: This includes the zeros in the columns.
        min_values = np.min(matrix, axis=0)

        # Iterate through all of the columns and rows and subtract the column-respective min_value from all of the
        # numbers in the respective column. The subtraction throughout the matrix rows yields a column reduced matrix.
        for column in range(len(matrix[0])):
            for row in range(len(matrix)):
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[column]

        return matrix, min_values

    # Sequential row to column reductions yield a completely reduced matrix.
    matrix, min_row = row_reduction()
    matrix, min_col = col_reduction()

    # Iterate over the min_row and min_column lists and add each value within them to the cost variable.
    for index in range(len(min_row)):

        # If the element at the specified index is not infinity, add it to min_row and min_col.
        if min_row[index] != math.inf:
            cost += min_row[index]
        if min_col[index] != math.inf:
            cost += min_col[index]

    # Return the cost and matrix.
    return cost, matrix


def explore_edge(start_node, node_from, node_to, matrix):
    """Edge modification on exploration.

    Sets the node_from row, node_to column, and the point (node_to, node_from) to math.inf. Returns the resulting
    matrix. The edges explored get set to math.inf.

    :params: start_node: The node starting node.
    :params: node_from: The node from which the edge is being explored from (row).
    :params: node_to: The node to which the edge is being (column).
    :params: matrix: The matrix to modify as a result of the exploration.
    :returns: matrix: A modified matrix that has been explored according to the input parameters.

    """

    # Set rows to math.inf.
    matrix[node_from, :] = math.inf

    # Set columns to math.inf.
    matrix[:, node_to] = math.inf

    # Set (j,i) to math.inf.
    matrix[node_to, start_node] = math.inf

    return matrix


def search_solid_state_tree(root):
    """Print items from the Solid State Tree.

    Method to search and print everything from Solid State tree.

    :params: node_object root: Parent node from the solid state tree.

    """

    if root.parent_node is not None:
        print("Parent: ", root.parent_node.node_id, "ID: ", root.node_id, "House: ", root.node.id)

    else:
        print("ID: ", root.node_id, "House: ", root.node.id)

    for child in root.children:
        search_solid_state_tree(child)


def heuristic(long1, lat1, long2, lat2, include_elevation, elevation_dict=None):
    """Determine the euclidean distance.

    Determine the euclidean distance between a pair of start and end coordinates. Optional: Account for the elevation
    when calculating the distance.

    :params: int long1: Starting longitude
    :params: int lat1: Starting latitude
    :params: int long2: End longitude
    :params: int lat2: End latitude
    :params: boolean include_elevation: Whether to account for the elevation or not.
    :params: dict elevation_dict: Dictionary containing the node longitude/latitude as the keys, and elevation as the
                                  value.
    :returns: cost: Euclidean distance between the start and end points.

    """

    if include_elevation:
        elevation1 = elevation_dict[long1, lat1] / 5500
        elevation2 = elevation_dict[long2, lat2] / 5500
        return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2 + (elevation2 - elevation1) ** 2)

    return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2)


# Algorithms
def node_to_node_search(start_node, goal, intersection_dict, elevation_dict):
    """A* algorithm.

    Find the shortest euclidean distance between any two nodes.

    :params: int start_node: The initial node where the algorithm should begin from.
    :params: int goal: The goal node where the algorithm should end at.
    :params: dict intersection_dict: Dictionary containing the longitude/latitude as the keys and the values as the
                                     respective index number for all the intersection nodes.
    :params: elevation_dict: Dictionary containing the node longitude/latitude as the keys, and elevation as the value.

    """

    # This is the A* search algorithm, it using the euclidean distance from the goal as a heuristic to try given nodes.
    open_list = [start_node]
    closed_list = []
    final_route = []

    start_node.f = 0
    start_node.g = 0
    tree_dict = {start_node: 0}

    while len(open_list) != 0:
        current_node = open_list[0]

        # Set the new node to be the previous sub_node with the lowest f(x).
        if current_node != start_node:
            current_node = min(open_list, key=attrgetter('f'))

        # Level represents the node.
        level = tree_dict[current_node]
        open_list.remove(current_node)

        if level == len(final_route):
            final_route.append(current_node)
        else:
            final_route[level] = current_node

        # Add the current node to the checked list
        closed_list.append(current_node)

        # Check if the goal node has been reached
        if current_node == goal:
            return current_node.f

        # For the current_node, go through all of its connections, and determine their f(x) as a sum of their h(x)
        # and g(x).
        for sub_node in current_node.connections:
            if sub_node in intersection_dict.keys():
                sub_node = intersection_dict[sub_node]
            else:
                continue
            if sub_node in closed_list:
                continue
            found = False
            for key, value in tree_dict.items():
                if key == sub_node:
                    found = True
            if not found:
                tree_dict[sub_node] = level + 1

            sub_node.g = heuristic(current_node.long_lat[0], current_node.long_lat[1], sub_node.long_lat[0],
                                   sub_node.long_lat[1], True, elevation_dict) + current_node.g
            sub_node.h = heuristic(sub_node.long_lat[0], sub_node.long_lat[1], goal.long_lat[0],
                                   goal.long_lat[1], True, elevation_dict)
            sub_node.f = sub_node.g + sub_node.h

            open_list.append(sub_node)


def space_state_algorithm(start_node, original_matrix, graph):
    """Space State algorithm.

    Generate a space state tree to determine the optimal route through the generated network.

    :params: int start_node: The initial node where the algorithm should begin from.
    :params: list original_matrix: 2D list consisting of the edge weights between V*V nodes.
    :params: graph_object graph: Graph of the network.
    :returns: final_list, visual_list: A list containing the path to be taken to achieve the lowest cost route through
                                       the network. A second list consisting of the start node, end node, and the edge
                                       weight between the start and end node.

    """

    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix, this will give us an estimate of the cost of the root, and then then by reducing
    # the roots matrix we will get a cost estimate for other nodes.
    initial_cost, cost_matrix = calculate_cost(original_matrix_copy)

    # Create variables
    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []
    visual_list = []
    final_list = []

    root = SolidStateNode(None, 0, parent_node, 0, cost_matrix, initial_cost)
    SolidStateTree(root)

    priority_queue.push_wo_matrix(root, initial_cost)

    i = 1
    num_of_backtracks = 0

    lowest_branch_cost = math.inf
    lowest_branch = []
    first_branch = []
    first_branch_cost = math.inf
    return_val = []

    # Priority queue takes the root, and then it takes all other nodes that have been discovered from the root.
    while not priority_queue.isEmpty():

        parent_state, cost = priority_queue.pop_wo_matrix()

        print("First branch: ", first_branch)
        print("Lowest branch: ", lowest_branch)

        parent = parent_state.node
        parent_matrix = parent_state.reduced_matrix
        # Check to see if a branch has been completed, if so, establish it as the "first branch."
        if parent_state.level >= graph.size() - 1 + num_of_backtracks:
            if len(first_branch) == 0:
                first_branch_cost = parent_state.cost
                next_state = parent_state
                while next_state is not None:
                    first_branch.append(next_state.node.id)
                    next_state = next_state.parent_node
                first_branch.reverse()
                # lowest branch is currently the first_branch.
                lowest_branch = first_branch
                lowest_branch_cost = first_branch_cost

            root_connection = False
            # Check to see if there is a root connection. If so, we will link it into a circle.
            for connection in parent.connections:
                if connection.id == root.node.id:
                    root_connection = True

            if not root_connection:
                continue

            # Deepcopy parent matrix so we don't have to worry about pointers.
            parent_matrix_copy = copy.deepcopy(parent_matrix)

            parent_copy_explored = explore_edge(parent.id, parent.id, root.node.id, parent_matrix_copy)
            # Further reduce matrix to estimate cost.
            cost_for_step, parent_copy_reduced = calculate_cost(parent_copy_explored)

            branch_cost = cost_for_step
            branch_cost += cost
            branch_cost += cost_matrix[parent.id, root.node.id]

            if not priority_queue.isEmpty():
                # Peek at next node, see if the next nodes cost is less than the lowest branch's cost. IF so,
                # we return the branch. Otherwise, we continue searching for "better" branches.
                next_node, next_node_cost = priority_queue.peek()

                if branch_cost <= next_node_cost:
                    if branch_cost < first_branch_cost * 2:
                        while parent_state is not None:
                            return_val.append(parent_state.node.id)
                            parent_state = parent_state.parent_node
                        return_val.reverse()
                        return_val.append(0)
                        break

                    else:
                        return_val = first_branch
                        break
            # If priority queue is empty, return the lowest branch.
            if priority_queue.isEmpty():
                if first_branch_cost * 2 < lowest_branch_cost:
                    return_val = first_branch
                else:
                    return_val = lowest_branch_cost

            # If lowest branch is lower than current branch, make current branch the lowest branch and return.
            if lowest_branch_cost > branch_cost:
                lowest_branch_cost = branch_cost
                lowest_branch = []
                lowest_state_node = parent_state
                lowest_node = parent
                while lowest_state_node is not None:
                    lowest_branch.append(lowest_state_node.node.id)
                    lowest_state_node = lowest_state_node.parent_node

                lowest_branch.reverse()
                lowest_branch.append(0)
                continue

        # If current cost is greater than lowest branch, return the lowest branch.
        if cost > lowest_branch_cost:
            return_val = lowest_branch
            break
        for sub_node in parent.connections:
            if sub_node not in parent_state.closedList or len(parent.connections) == 1:
                # Set initial cost to that of the parent cost.
                total_cost = cost
                parent_matrix_copy = copy.deepcopy(parent_matrix)

                # Add the cost of the edge to the total_cost. The edge cost must be retrieved from the starting node
                # reduced cost matrix (cost_matrix).
                edge_cost = cost_matrix[parent.id, sub_node.id]
                total_cost += edge_cost

                # Add the cost of the lower bound starting at the sub_node. This means that we need to add the
                # math.inf values, then perform a row and column reduction, and add the resulting cost to our total
                # cost. NOTE: This must be done on the current parent reduced matrix.
                parent_matrix_copy_explored = explore_edge(start_node, parent.id, sub_node.id, parent_matrix_copy)
                cost_for_step, parent_matrix_copy_explored_reduced = calculate_cost(parent_matrix_copy_explored)
                total_cost += cost_for_step

                sub_node.cost = total_cost

                state_node = SolidStateNode(parent_state, i, sub_node, parent_state.level + 1,
                                            parent_matrix_copy_explored, total_cost)
                i += 1

                priority_queue.push_wo_matrix(state_node, cost)
        closed_list.append(parent)
    closed_list = lowest_branch

    # Once we have a value to return, we append it to a final list and create a list in the correct style for
    # visualization.
    if len(return_val) > 0:
        parent_index = 0

        for item in return_val:
            final_list.append(item)

            if len(final_list) >= 2:
                visual_list.append(
                    [final_list[parent_index], item, original_matrix[final_list[parent_index]][item]])
                parent_index += 1

    else:
        matrix_insert = 0
        for item in closed_list:
            if len(closed_list) - len(final_list) < len(closed_list):
                visual_list.append(
                    [final_list[matrix_insert], item, original_matrix[final_list[matrix_insert]][item]])
                matrix_insert += 1

            final_list.append(item)

            if len(closed_list) - len(final_list) == 0:
                visual_list.append(
                    [final_list[matrix_insert], item, original_matrix[final_list[matrix_insert]][item]])

    return final_list, visual_list


def prims_algorithm(start_node, original_matrix, graph):
    """Prims algorithm.

    Generate a minimum spanning tree via Prims algorithm based on the graph data.

    :params: int start_node: The initial node where the algorithm should begin from.
    :params: list original_matrix: 2D list consisting of the edge weights between V*V nodes.
    :params: graph_object graph: Graph of the network.
    :returns: visual_list: A list consisting of the start node, end node, and the edge weight between the start and end
                           node.

    """

    # Make a copy of the original matrix, this is needed for the Prims algorithm.
    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    visual_list = []

    priority_queue.push_with_matrix(parent_node, initial_cost, cost_matrix)

    # A Python program for Prims' Minimum Spanning Tree (MST) algorithm.
    # The program is for adjacency matrix representation of the graph

    class PrimsGraph:

        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

        # A utility function to print the constructed MST stored in parent[]
        def printMST(self, parent):
            print("Edge \tWeight")
            for i in range(1, self.V):
                print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
                visual_list.append([parent[i], i, self.graph[i][parent[i]]])

        # A utility function to find the vertex with minimum distance value, from the set of vertices
        # not yet included in shortest path tree
        def minKey(self, key, mstSet):
            # Initialize min value
            minimum = float("inf")
            min_index = 0

            for v in range(self.V):
                if key[v] < minimum and mstSet[v] == False:
                    minimum = key[v]
                    min_index = v

            return min_index

        # Function to construct and print MST for a graph represented using adjacency matrix representation
        def primMST(self):
            # Key values used to pick minimum weight edge in cut
            key = [float("inf")] * self.V

            # Array to store constructed MST
            parent = [None] * self.V
            # Make key 0 so that this vertex is picked as first vertex
            key[0] = start_node
            mstSet = [False] * self.V

            # First node is always the root of
            parent[0] = -1

            for _ in range(self.V):
                # Pick the minimum distance vertex from the set of vertices not yet processed.
                # u is always equal to src in first iteration
                u = self.minKey(key, mstSet)

                # Put the minimum distance vertex in
                # the shortest path tree
                mstSet[u] = True

                # Update dist value of the adjacent vertices of the picked vertex only if the current
                # distance is greater than new distance and the vertex in not in the shortest path tree
                for v in range(self.V):

                    # graph[u][v] is non zero only for adjacent vertices of m
                    # mstSet[v] is false for vertices not yet included in MST
                    # Update the key only if graph[u][v] is smaller than key[v]
                    if 0 < self.graph[u][v] < key[v] and mstSet[v] == False:
                        key[v] = self.graph[u][v]
                        # print(u, v, self.graph[u][v])
                        parent[v] = u

            self.printMST(parent)

    g = PrimsGraph(len(original_matrix_copy[0]))
    g.graph = original_matrix_copy

    g.primMST()

    print("Prims algorithm doesn't work in the typically sense. So, the comparison to the ideal path won't be "
          "successful.")
    return visual_list, visual_list


def visualization(graph_data=None, path=None):
    """Visualize the original and resulting network.

    Creates a visualization of the original network and the network that the algorithms deem to be ideal.

    :params: graph_object graph_data: Graph object that contains all of the nodes and their respective edge costs.
    :params: list path: List containing the starting node, ending node, and cost for all nodes in the graph.

    """

    if graph_data is not None:
        plt.figure(1)
        G = nx.Graph()

        # List of the node ids.
        node_list = []

        # List of the node-node connections.
        edge_list = []

        # Dictionary containing the node-node connections and their respective costs.
        edge_labels_dict = {}

        for node in graph_data.nodes:
            node_list.append(node.id)
            for connections in node.connections:
                edge_list.append([node.id, connections.id])

                # Multiplying by 1000 makes the weights more understandable at a first glance. The rounding is present
                # as the decimal value is quite long.
                edge_labels_dict[(node.id, connections.id)] = round(node.connections[connections] * 1000, 2)

        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        pos = nx.spring_layout(G)

        nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1", node_color='orange', font_color='white')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict)

    if path is not None:
        plt.figure(2)
        H = nx.Graph()

        # List of the node ids.
        node_list = []

        # List of the node-node connections.
        edge_list = []

        # Dictionary containing the node-node connections and their respective costs.
        edge_labels_dict = {}

        for row in path:
            parent_node = row[0]
            sub_node = row[1]

            # Multiplying by 1000 makes the weights more understandable at a first glance. The rounding is present as
            # the decimal value is quite long.
            edge_weight = round(row[2] * 1000, 2)

            node_list.append(parent_node)
            edge_list.append([parent_node, sub_node])

            edge_labels_dict[(parent_node, sub_node)] = edge_weight

        H.add_nodes_from(node_list)
        H.add_edges_from(edge_list)

        pos = nx.spring_layout(H)

        nx.draw(H, pos, with_labels=True, connectionstyle="arc3,rad=0.1", node_color='orange', font_color='white')
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels_dict)

    # Generate and show the selected plot(s).
    plt.show()


def main():
    # Fetch CSV files, load all DataFrames from csv files.
    if LOAD_DATA:
        # Read the zeroth and first CV files.
        road_line_nodes = pd.read_csv("0_processed_road_lines", sep=',')
        node_data_wo_intersections = pd.read_csv("1_processed_road_line_nodes", sep=',')

        # Process the node_data_wo_intersections.
        node_coord_pairs, multi_index_pairs = node_dictionary(node_data_wo_intersections)

        # Read the third CV files.
        node_data = pd.read_csv("2_processed_road_line_nodes_w_intersections", sep=',')

        # Load the elevation_dict using pickle, so that we don't need to reprocess it.
        with open("elevation_1.pkl", "rb") as tf:
            elevation_dict = pickle.load(tf)

        # Load graphs from pickle, so that we don't need to reprocess it.
        with open("graph_1.pkl", "rb") as tf:
            graph = pickle.load(tf)

        with open("nodes_for_graph_1.pkl", "rb") as tf:
            nodes_for_graph = pickle.load(tf)

        nodes_list = graph.keys()

    # Otherwise, fetch the road lines and nodes from the given .shp files, load them, and store them as csv files to
    # increase future processing speed.
    else:
        road_line_nodes = road_line_processing(ROAD_LINE_DATA)
        road_line_nodes.to_csv("0_processed_road_lines", encoding='utf-8', index=False)

        node_data = line_node_dataframe(road_line_nodes)
        node_data.to_csv("1_processed_road_line_nodes", encoding='utf-8', index=False)

        node_coord_pairs, multi_index_pairs = node_dictionary(node_data)

        node_data = intersection_node_check(node_data, node_coord_pairs, multi_index_pairs)
        node_data.to_csv("2_processed_road_line_nodes_w_intersections", encoding='utf-8', index=False)

        elevation_dict = road_elevation_processing(ROAD_POINT_WITH_ELEVATION_DATA)

        with open("elevation_1.pkl", "wb") as fp:
            pickle.dump(elevation_dict, fp)

        # Re-index node data dataframe.
        node_data.reset_index().drop("index", axis=1)
        node_data = node_data.reset_index()
        node_data = node_data.drop("index", axis=1)

        # Drop column two, which kept track fo duplicates from dataframe.
        node_data = node_data.drop(2, axis=1)

        node_data = node_data.rename(
            columns={'0': "Long/Lat Coordinates", '1': "Connections", '3': "Is Intersection", })

        intersection_dict = intersection_node_dictionary(node_data)

        # Get houses and intersections.
        house_ids = get_houses(10)
        nearest_intersections = nearest_intersection_to_house(house_ids, node_data, intersection_dict)
        graph = find_nearby_houses(nearest_intersections, intersection_dict, elevation_dict)

        nodes_list = graph.keys()

        nodes_for_graph = []
        id_num = 0

        # Append nodes to graph.
        for node in nodes_list:
            new_node = Node(None, None, node.long_lat, None, id_num)
            nodes_for_graph.append(new_node)
            id_num += 1

        # Dump the nodes_for_graph and graph into pickle files to save for later.
        with open("nodes_for_graph_1.pkl", "wb") as fp:
            pickle.dump(nodes_for_graph, fp)

        with open("graph_1.pkl", "wb") as fp:
            pickle.dump(graph, fp)

    random.seed(42)

    # Make sure that nodes are connected to other nearby houses.
    for graph_node in nodes_for_graph:
        graph_node.convert_connections_for_graph()
        for node in nodes_list:
            if node.long_lat == graph_node.long_lat:
                for connection in graph[node]:
                    for other_graph_node in nodes_for_graph:
                        if connection[0].long_lat == other_graph_node.long_lat:
                            # Multiply connection by randomized priority score.
                            graph_node.connections[other_graph_node] = connection[1] * randint(1, 5) * 0.001

    # Force graph to be bidirectional.
    for graph_node in nodes_for_graph:
        for _ in graph_node.connections:
            for other_graph_node in nodes_for_graph:
                if other_graph_node in graph_node.connections.keys() and graph_node not in other_graph_node.connections.keys():
                    other_graph_node.connections[graph_node] = graph_node.connections[other_graph_node]

    graphical_data = Graph(nodes_for_graph)
    matrix = graphical_data.convert_to_matrix()

    # Choose the algorithm to run, where 1 is the space state and any other value is prims.
    if ALGORITHM == 0:
        final_path, visual_path = space_state_algorithm(0, matrix, graphical_data)
    else:
        final_path, visual_path = prims_algorithm(0, matrix, graphical_data)

    # Visualization of the original network and the ideal network.
    if VISUALIZATION:
        visualization(graph_data=graphical_data, path=visual_path)

    print("\n------------------------")
    print("Minutes since execution:", (time.time() - START_TIME) / 60)


if __name__ == "__main__":
    main()
