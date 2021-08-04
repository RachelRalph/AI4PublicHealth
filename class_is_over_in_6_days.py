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
from operator import attrgetter
import matplotlib.pyplot as plt
import networkx as nx
import time
import pickle

# Get the start time.
START_TIME = time.time()
LOAD_DATA = False

# # Get script and dataset file paths.
SCRIPT_PATH = os.path.dirname(__file__)

# Read the building point data w/ elevation .shp file via geopandas and store as a pandas dataframe.
BUILDING_FILE_WITH_ELEVATION = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_buildings_with_elevation.shp")
BUILDING_FILE_WITH_ELEVATION = gpd.read_file(BUILDING_FILE_WITH_ELEVATION)
BUILDING_FILE_WITH_ELEVATION = pd.DataFrame(BUILDING_FILE_WITH_ELEVATION)

if not LOAD_DATA:
    # Read the road line data .shp file via geopandas and store as a pandas dataframe.
    ROAD_LINE_DATA = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_lines_CORRECT.shp")
    ROAD_LINE_DATA = gpd.read_file(ROAD_LINE_DATA)
    ROAD_LINE_DATA = pd.DataFrame(ROAD_LINE_DATA)

    # Read the road points w/ elevation .shp file via geopandas and store as a pandas dataframe.
    ROAD_POINT_WITH_ELEVATION_PATH = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_pointdata_with_elevation.shp")
    ROAD_POINT_WITH_ELEVATION_DATA = gpd.read_file(ROAD_POINT_WITH_ELEVATION_PATH)
    ROAD_POINT_WITH_ELEVATION_DATA = pd.DataFrame(ROAD_POINT_WITH_ELEVATION_DATA)

# Data processing

"""
Note on Data preprocessing: the road node data in OpenStreetMaps is unusable because the bearings and distances of 
each node are incredibly unreliable. The line data is much better, however for our algorithm to work, 
we must convert the line data into node data. Below is our code to process each road into a list of coordinates, 
And from there we use the roads to deduce each nodes connections (the nodes next to them on the roads), 
and which nodes are intersections. 
"""
def road_line_processing(road_line_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    processed_data_line = []

    # Stores Coordinates from geometry.
    for index, rows in road_line_df.iterrows():
        coordinates_line = road_line_df.iloc[index, 11]
        string_type = (type(coordinates_line))

        if str(string_type) == "<class 'shapely.geometry.linestring.LineString'>":
            coordinates_line = list(coordinates_line.coords)

            start_longitude_line = round(coordinates_line[0][0], 6)
            start_latitude_line = round(coordinates_line[0][1], 6)
            end_longitude_line = round(coordinates_line[-1][0], 6)
            end_latitude_line = round(coordinates_line[-1][1], 6)

            all_coord = []

            # Appends all coordinates from line_string into a list
            for coordinate in coordinates_line:
                all_coord.append(coordinate)

            processed_data_line.append(
                (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line, all_coord))

    # Moves into a dataframe with the start coordinates of every road, and a coordinate list containing all coordinates.
    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(
        columns={0: "Start Longitude", 1: "Start Latitude", 2: "End Longitude", 3: "End Latitude",
                 4: "Coordinates List"})

    print("Minutes since execution:", (time.time() - START_TIME) / 60)  # 0.03

    # Return data preprocessing line.
    return processed_data_line


def line_node_dataframe(road_line_df):
    """This function creates a dataframe where each set of coordinates has it's own row, with the coordinates that
    it's connecting to (or next to on the road) """

    node_information = []

    for index, row in road_line_df.iterrows():
        for i in range(len(row["Coordinates List"])):
            connections = []
            coordinate_pair = row["Coordinates List"][i]

            if i != 0:
                connections.append(row["Coordinates List"][i - 1])

            if i != len(row["Coordinates List"]) - 1:
                connections.append(row["Coordinates List"][i + 1])
            node_information.append([coordinate_pair, connections])

    node_data = pd.DataFrame(node_information)

    print("Minutes since execution:", (time.time() - START_TIME) / 60)  # 0.04

    return node_data

test = 1

radius_earth = 6378.137

def road_elevation_processing(road_elevation_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    #Create a dictionary that contains latitude, longtitude coordinates and the corresponding elevation.
    elevation_dict = {}

    
    for rows in range(len(road_elevation_df.index)):
        
        # Get the coordinates.
        coordinates = list(road_elevation_df.iloc[rows, 22].coords)
        start_latitude = coordinates[0][1] # Contains max 7 decimal points
        start_longitude = coordinates[0][0] # Contains max 7 decimal points

        # Get the elevation, road condition, and road type.
        elevation = road_elevation_df.iloc[rows, 17]
        

        elevation_dict[(start_longitude, start_latitude)] = elevation

    
    return elevation_dict

"""
This takes the above dataframe and converts it into a dictionary where the latitude/longitude pairs are the keys, 
in order to decrease runtime. This function also logs all nodes it processes more that once in the multiple index 
pairs list. Because a node was processed more than once, it must include more than one row, ergo, 
multiple_index_pairs flags all the nodes that are intersections. 
"""
def node_dictionary(node_df):
    long_lat_pairs = {}

    multiple_index_pairs = []


    
    for index, row in node_df.iterrows():
        long_lat_coord = row[0]

        if long_lat_coord in long_lat_pairs:
            long_lat_pairs[long_lat_coord].append(index)

            if long_lat_coord not in multiple_index_pairs:
                multiple_index_pairs.append(long_lat_coord)

        else:
            long_lat_pairs[long_lat_coord] = [index]

    print("Minutes since execution:", (time.time() - START_TIME) / 60)  # 0.3

    return long_lat_pairs, multiple_index_pairs


"""
Sets an IsIntersection value in the dataframe for all intersections. Furthermore, a thrid value is set. This value 
is set for nodes that where processed more than once; and have an equivelent node flagged as an intersection in the
dataframe. this happens so that we know which nodes we don't have to process more than once. 
"""
def intersection_node_check(node_df, node_dict, index_pairs):
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

    print("Minutes since execution:", (time.time() - START_TIME) / 60)  # 2.6

    return node_df


# Creates a dictionary of intersection nodes, for the sake of easy look up.
def intersection_node_dictionary(node_df):
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

    print("Minutes since execution:", (time.time() - START_TIME) / 60)

    return intersection_node_dict


# Get random houses
def get_houses(number_of_houses):
    house_ids = []

    # Seed can be changed to get different random numbers.
    random.seed(42)

    for i in range(number_of_houses):
        house_ids.append(random.randint(0, len(BUILDING_FILE_WITH_ELEVATION)))

    return house_ids


# Get nearest intersection for houses, to use as a benchmark for algorithms below. 
def nearest_intersection_to_house(houses, node_data, intersection_dict):
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

    return nearest_intersections


# This function finds the three nearest houses to perfrom an A* search on, so that we can find the quickest path
# between house A and house B.
def find_nearby_houses(nearest_intersections, intersection_dict, elevation_dict):
    graph = {}

    for intersection in nearest_intersections:
        houses_completed = []
        houses_visited = []

        for i in range(3):
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

    # Function to convert the graph into an adjancy matrix. Where there are no connections, the value of the matrix
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
    """Calculates the reduction cost of the matrix. Returns the adjacency matrix (reduced matrix) and the associated
    cost. """
    cost = 0

    def row_reduction():
        """Reduce the rows. Each row is reduced based on the lowest number in itself. Returns the row reduced matrix."""

        # Get minimum values of all the rows. Note that this includes the zeros in the rows.
        min_values = np.min(matrix, axis=1)

        # Iterate through all of the rows and columns and subtract the row-respective min_value from all of the
        # numbers in the respective row.
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[row]

        return matrix, min_values

    def col_reduction():
        """Reduce the columns. Each column is reduced based on the lowest number in itself. Returns the column
        reduced matrix. """

        # Get minimum values of all the columns. Note that this includes the zeros in the columns.
        min_values = np.min(matrix, axis=0)

        # Iterate through all of the columns and rows and subtract the column-respective min_value from all of the
        # numbers in the respective column.
        for column in range(len(matrix[0])):
            for row in range(len(matrix)):
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[column]

        return matrix, min_values

    matrix, min_row = row_reduction()
    matrix, min_col = col_reduction()

    # Iterate over the min_row and min_column lists and add each value within them to the cost variable.
    for index in range(len(min_row)):
        if min_row[index] != math.inf:
            cost += min_row[index]
        if min_col[index] != math.inf:
            cost += min_col[index]

    return cost, matrix


def explore_edge(start_node, node_from, node_to, matrix):
    """Sets the node_from row, node_to column, and the point (node_to, node_from) to math.inf. Returns the resulting
    matrix. """

    # Set rows to math.inf.
    matrix[node_from, :] = math.inf

    # Set columns to math.inf.
    matrix[:, node_to] = math.inf

    # Set (j,i) to math.inf.
    matrix[node_to, start_node] = math.inf

    return matrix


def search_solid_state_tree(root):
    # Method to search and print everything from Solid State tree.
    if root.parent_node is not None:
        print("Parent: ", root.parent_node.node_id, "ID: ", root.node_id, "House: ", root.node.id)

    else:
        print("ID: ", root.node_id, "House: ", root.node.id)

    for child in root.children:
        search_solid_state_tree(child)


def heuristic(long1, lat1, long2, lat2, include_elevation, elevation_dict=None):

    if include_elevation:
        elevation1 = elevation_dict[long1, lat1]/5500
        elevation2 = elevation_dict[long2, lat2]/5500
        return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2 + (elevation2 - elevation1)**2)
  
    return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2)
        
# Algorithms
def node_to_node_search(start_node, goal, intersection_dict, elevation_dict):
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
            long_lat = sub_node
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


# Branch and bound algorithm.
def space_state_algorithm(start_node, original_matrix, graph):
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
    # Make a copy of the original matrix, this is needed for the Prims algorithm.
    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    visual_list = []

    priority_queue.push_with_matrix(parent_node, initial_cost, cost_matrix)

    # A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
    # The program is for adjacency matrix representation of the graph

    class PrimsGraph:

        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for _ in range(vertices)]
                          for _ in range(vertices)]

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
                """print("\nParent: ", parent)
                print("Keys: ", key)
                print("mstSet: ", mstSet)"""
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
    if graph_data is not None:
        plt.figure(1)
        G = nx.Graph()

        node_list = []
        edge_list = []
        edge_labels_dict = {}

        for node in graph_data.nodes:
            node_list.append(node.id)
            for connections in node.connections:
                edge_list.append([node.id, connections.id])
                edge_labels_dict[(node.id, connections.id)] = node.connections[connections]

        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        pos = nx.spring_layout(G)

        nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1", node_color='orange', font_color='white')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict)
        # plt.savefig("simple_path.png")  # save as png

    if path is not None:
        plt.figure(2)
        H = nx.Graph()

        node_list = []
        edge_list = []
        edge_labels_dict = {}

        for row in path:
            parent_node = row[0]
            sub_node = row[1]
            edge_weight = row[2]

            node_list.append(parent_node)
            edge_list.append([parent_node, sub_node])

            edge_labels_dict[(parent_node, sub_node)] = edge_weight

        print(node_list)
        print(edge_list)
        print(edge_labels_dict)

        H.add_nodes_from(node_list)
        H.add_edges_from(edge_list)

        pos = nx.spring_layout(H)

        nx.draw(H, pos, with_labels=True, connectionstyle="arc3,rad=0.1", node_color='orange', font_color='white')
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels_dict)
        # plt.savefig("simple_path.png")  # save as png

    plt.show()


def main():
    VISUALIZATION = True
    ALGORITHM = 1

    # Fetch csv files, load all dataframes from csv files.
    if LOAD_DATA:
        road_line_nodes = pd.read_csv("0_processed_road_lines", sep=',')
        node_data_wo_intersections = pd.read_csv("1_processed_road_line_nodes", sep=',')
        node_coord_pairs, multi_index_pairs = node_dictionary(node_data_wo_intersections)
        node_data = pd.read_csv("2_processed_road_line_nodes_w_intersections", sep=',')

        # Load graphs from pickle, so that we don't have to reorganize it.
        with open("graph_1.pkl", "rb") as tf:
            graph = pickle.load(tf)

        with open("nodes_for_graph_1.pkl", "rb") as tf:
            nodes_for_graph = pickle.load(tf)

        with open("elevation_1.pkl", "rb") as tf:
            nodes_for_graph = pickle.load(tf)

        nodes_list = graph.keys()

    else:
        # Otherwise we get the road lines and nodes from the given sahpe files, load them, and store them as csvs.
        road_line_nodes = road_line_processing(ROAD_LINE_DATA)
        road_line_nodes.to_csv("0_processed_road_lines", encoding='utf-8', index=False)

        node_data = line_node_dataframe(road_line_nodes)
        node_data.to_csv("1_processed_road_line_nodes", encoding='utf-8', index=False)

        node_coord_pairs, multi_index_pairs = node_dictionary(node_data)

        node_data = intersection_node_check(node_data, node_coord_pairs, multi_index_pairs)
        node_data.to_csv("2_processed_road_line_nodes_w_intersection", encoding='utf-8', index=False)


        elevation_dict = road_elevation_processing(ROAD_POINT_WITH_ELEVATION_DATA)

        # re-index node data dataframe.

        node_data.reset_index().drop("index", axis=1)
        node_data = node_data.reset_index()
        node_data = node_data.drop("index", axis=1)

        # Drop column two, which kept track fo duplicates from dataframe.

        if LOAD_DATA:
            node_data = node_data.drop('2', axis=1)
        else:
            node_data = node_data.drop(2, axis=1)

        node_data = node_data.rename(
            columns={'0': "Long/Lat Coordinates", '1': "Connections", '3': "Is Intersection", })

        intersection_dict = intersection_node_dictionary(node_data)

        # Get houses and intersections.

        house_ids = get_houses(10)
        nearest_intersections = nearest_intersection_to_house(house_ids, node_data, intersection_dict)
        graph = find_nearby_houses(nearest_intersections, intersection_dict, elevation_dict)
        print(graph)

        nodes_list = graph.keys()

        nodes_for_graph = []
        id_num = 0

        # Append nodes to grpah.

        for node in nodes_list:
            new_node = Node(None, None, node.long_lat, None, id_num)
            connections = {}
            nodes_for_graph.append(new_node)
            id_num += 1

        # Dump them into a pickle file to save for later.
        with open("nodes_for_graph_1.pkl", "wb") as fp:
            pickle.dump(nodes_for_graph, fp)

        with open("graph_1.pkl", "wb") as fp:
            pickle.dump(graph, fp)

        with open("elevation_1.pkl", "wb") as fp:
            pickle.dump(elevation_dict, fp)

    random.seed(42)
    # Make sure that nodes are connected to other nearby houses.
    for graph_node in nodes_for_graph:
        graph_node.convert_connections_for_graph()
        for node in nodes_list:
            if node.long_lat == graph_node.long_lat:
                for connection in graph[node]:
                    for other_graph_node in nodes_for_graph:
                        if connection[0].long_lat == other_graph_node.long_lat:
                            graph_node.connections[other_graph_node] = connection[1]*rand_int(0, 5)*0.001 #Multiply connection by randomized priority score. 

    # Force graph to be bidirectional.

    for graph_node in nodes_for_graph:
        for connection in graph_node.connections:
            for other_graph_node in nodes_for_graph:
                if other_graph_node in graph_node.connections.keys() and graph_node not in other_graph_node.connections.keys():
                    other_graph_node.connections[graph_node] = graph_node.connections[other_graph_node]

    graphical_data = Graph(nodes_for_graph)

    matrix = graphical_data.convert_to_matrix()

    if ALGORITHM == 0:
        final_path, visual_path = space_state_algorithm(0, matrix, graphical_data)

    else:
        final_path, visual_path = prims_algorithm(0, matrix, graphical_data)

    if VISUALIZATION:
        visualization(graph_data=graphical_data, path=visual_path)

    print("\n------------------------")
    print("Minutes since execution:", (time.time() - START_TIME) / 60)


if __name__ == "__main__":
    main()
