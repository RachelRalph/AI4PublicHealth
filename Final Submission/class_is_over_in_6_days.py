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
LOAD_DATA = True

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
def road_line_processing(road_line_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    processed_data_line = []

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

            for coordinate in coordinates_line:
                all_coord.append(coordinate)

            processed_data_line.append(
                (start_longitude_line, start_latitude_line, end_longitude_line, end_latitude_line, all_coord))

    processed_data_line = pd.DataFrame(processed_data_line)
    processed_data_line = processed_data_line.rename(
        columns={0: "Start Longitude", 1: "Start Latitude", 2: "End Longitude", 3: "End Latitude",
                 4: "Coordinates List"})

    print("Minutes since execution:", (time.time() - START_TIME) / 60)  # 0.03

    return processed_data_line


def line_node_dataframe(road_line_df):
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


# Data to dictionary and list
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


# Set boolean value for intersection nodes
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


def intersection_node_dictionary(node_df):
    intersection_node_dict = {}

    for index, row in node_df.iterrows():
        if row["Is Intersection"]:
            row_node = Node(None, None, eval(row["Long/Lat Coordinates"]), eval(row["Connections"]), None)

            if LOAD_DATA:
                intersection_node_dict[eval(row["Long/Lat Coordinates"])] = row_node

            else:
                intersection_node_dict[row["Long/Lat Coordinates"]] = row_node

    print("Minutes since execution:", (time.time() - START_TIME) / 60)

    return intersection_node_dict


# Get random houses
def get_houses(number_of_houses):
    house_ids = []

    random.seed(42)

    for i in range(number_of_houses):
        house_ids.append(random.randint(0, len(BUILDING_FILE_WITH_ELEVATION)))

    return house_ids


# Get nearest intersection for houses
def nearest_intersection_to_house(houses, node_data, intersection_dict):
    nearest_intersections = []

    for house_id in houses:
        coordinates = list(BUILDING_FILE_WITH_ELEVATION.loc[house_id, "geometry"].coords)
        long = coordinates[0][0]  # Contains max 7 decimal points
        lat = coordinates[0][1]  # Contains max 7 decimal points

        elevation = BUILDING_FILE_WITH_ELEVATION.at[house_id, "SAMPLE_1"]
        near_intersection = None
        near_intersection_dist = math.inf

        for index, row in node_data.iterrows():
            if row["Is Intersection"]:
                if LOAD_DATA:
                    node_long = eval(row["Long/Lat Coordinates"])[0]
                    node_lat = eval(row["Long/Lat Coordinates"])[1]

                else:
                    node_long = row["Long/Lat Coordinates"][0]
                    node_lat = row["Long/Lat Coordinates"][1]

                huer = heuristic(long, lat, node_long, node_lat)
                if near_intersection_dist > huer:
                    near_intersection_dist = huer
                    near_intersection = intersection_dict[(node_long, node_lat)]
        nearest_intersections.append(near_intersection)

    return nearest_intersections


def idk_what_this_does(nearest_intersections, intersection_dict):
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
                                     intersection.long_lat[1])
                    if dist < smallest_dist:
                        smallest_dist = dist
                        closest_house = house_to_go
            cost = node_to_node_search(closest_house, intersection, intersection_dict)
            houses_completed.append(closest_house)
            houses_visited.append((closest_house, cost))
            graph[intersection] = houses_visited

    return graph


# General classes
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

    def convert_connections_for_graph(self):
        self.connections = {}


class Graph:
    def __init__(self, nodes_for_graph):
        self.nodes = nodes_for_graph

    def convert_to_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        matrix = np.where(matrix == 0, math.inf, matrix)
        for node in self.nodes:
            for connections in node.connections:
                matrix[node.id][connections.id] = node.connections[connections]
        return matrix

    def size(self):
        return len(self.nodes)


class PriorityQueue:
    def __init__(self):
        self.queue = {}

    def isEmpty(self):
        return len(self.queue) == 0

    def push_with_matrix(self, node, priority, matrix):
        self.queue[node] = [priority, matrix]

    def push_wo_matrix(self, node, priority):
        self.queue[node] = priority

    def inQueue(self, node):
        for node in self.queue:
            if node.id == node.id:
                return True
        return False

    def pop_with_matrix(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node][0]
        matrix = self.queue[node][1]

        del self.queue[node]
        return node, cost, matrix

    def peek(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        return node, cost

    def pop_wo_matrix(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        del self.queue[node]
        return node, cost


class SolidStateNode:

    def __init__(self, parent_node, node_id, node, level, reduced_matrix, cost):
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

        # TODO: There has to be a more efficient way of doing this than going through two for loops.
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

        # TODO: There has to be a more efficient way of doing this than going through two for loops.
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
    if root.parent_node is not None:
        print("Parent: ", root.parent_node.node_id, "ID: ", root.node_id, "House: ", root.node.id)

    else:
        print("ID: ", root.node_id, "House: ", root.node.id)

    for child in root.children:
        search_solid_state_tree(child)


def heuristic(long1, lat1, long2, lat2):
    return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2)


# Algorithms
def node_to_node_search(start_node, goal, intersection_dict):
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

        #
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
                                   sub_node.long_lat[1]) + current_node.g
            sub_node.h = heuristic(sub_node.long_lat[0], sub_node.long_lat[1], goal.long_lat[0],
                                   goal.long_lat[1]) * 4146.282847732093
            sub_node.f = sub_node.g + sub_node.h

            open_list.append(sub_node)


def space_state_algorithm(start_node, original_matrix, graph):
    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix_copy)

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

    while not priority_queue.isEmpty():

        parent_state, cost = priority_queue.pop_wo_matrix()

        print("First branch: ", first_branch)
        print("Lowest branch: ", lowest_branch)

        parent = parent_state.node
        parent_matrix = parent_state.reduced_matrix
        if parent_state.level >= graph.size() - 1 + num_of_backtracks:
            if len(first_branch) == 0:
                first_branch_cost = parent_state.cost
                next_state = parent_state
                while next_state is not None:
                    first_branch.append(next_state.node.id)
                    next_state = next_state.parent_node
                first_branch.reverse()
                lowest_branch = first_branch

            root_connection = False
            for connection in parent.connections:
                if connection.id == root.node.id:
                    root_connection = True

            if not root_connection:
                continue

            parent_matrix_copy = copy.deepcopy(parent_matrix)

            parent_copy_explored = explore_edge(parent.id, parent.id, root.node.id, parent_matrix_copy)
            cost_for_step, parent_copy_reduced = calculate_cost(parent_copy_explored)

            branch_cost = cost_for_step
            branch_cost += cost
            branch_cost += cost_matrix[parent.id, root.node.id]

            if not priority_queue.isEmpty():
                next_node, next_node_cost = priority_queue.peek()

                if branch_cost <= next_node_cost:
                    if branch_cost < first_branch_cost * 2:
                        while parent_state is not None:
                            return_val.append(parent_state.node.id)
                            parent_state = parent_state.parent_node
                        # print("Return Value: ", return_val)
                        return_val.reverse()
                        return_val.append(0)
                        break

                    else:
                        return_val = first_branch
                        break

            if priority_queue.isEmpty():
                if first_branch_cost * 2 < lowest_branch_cost:
                    # print("Returning first branch...")
                    return_val = first_branch
                else:
                    return_val = lowest_branch

            if lowest_branch_cost > branch_cost:
                # print("Adding lowest branch...")
                lowest_branch_cost = branch_cost
                lowest_branch = []
                lowest_state_node = parent_state
                lowest_node = parent
                while lowest_state_node is not None:
                    lowest_branch.append(lowest_state_node.node.id)
                    lowest_state_node = lowest_state_node.parent_node

                lowest_branch.reverse()
                lowest_branch.append(0)

            # print("Going to the top of the loop...")

            continue

        if cost > lowest_branch_cost:
            return_val = lowest_branch
            # print("Found lowest cost")
            break
        # print("Going on down...")

        # if len(parent.connections) == 1:
        # num_of_backtracks = 1
        # print("Lowest Bracnh: ", lowest_branch)
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

            parent = [None] * self.V  # Array to store constructed MST
            # Make key 0 so that this vertex is picked as first vertex
            key[0] = start_node
            mstSet = [False] * self.V

            parent[0] = -1  # First node is always the root of

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
                # distance is greater than new distance and the vertex in not in the shotest path tree
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
    ALGORITHM = 0

    if LOAD_DATA:
        road_line_nodes = pd.read_csv("0_processed_road_lines", sep=',')
        node_data_wo_intersections = pd.read_csv("1_processed_road_line_nodes", sep=',')
        node_coord_pairs, multi_index_pairs = node_dictionary(node_data_wo_intersections)
        node_data = pd.read_csv("2_processed_road_line_nodes_w_intersections", sep=',')

        with open("graph_1.pkl", "rb") as tf:
            graph = pickle.load(tf)

        with open("nodes_for_graph_1.pkl", "rb") as tf:
            nodes_for_graph = pickle.load(tf)

        nodes_list = graph.keys()

    else:
        road_line_nodes = road_line_processing(ROAD_LINE_DATA)
        road_line_nodes.to_csv("0_processed_road_lines", encoding='utf-8', index=False)

        node_data = line_node_dataframe(road_line_nodes)
        node_data.to_csv("1_processed_road_line_nodes", encoding='utf-8', index=False)

        node_coord_pairs, multi_index_pairs = node_dictionary(node_data)

        node_data = intersection_node_check(node_data, node_coord_pairs, multi_index_pairs)
        node_data.to_csv("2_processed_road_line_nodes_w_intersection", encoding='utf-8', index=False)

        node_data.reset_index().drop("index", axis=1)
        node_data = node_data.reset_index()
        node_data = node_data.drop("index", axis=1)

        if LOAD_DATA:
            node_data = node_data.drop('2', axis=1)
        else:
            node_data = node_data.drop(2, axis=1)

        node_data = node_data.rename(
            columns={'0': "Long/Lat Coordinates", '1': "Connections", '3': "Is Intersection", })

        intersection_dict = intersection_node_dictionary(node_data)

        house_ids = get_houses(10)
        nearest_intersections = nearest_intersection_to_house(house_ids, node_data, intersection_dict)
        graph = idk_what_this_does(nearest_intersections, intersection_dict)

        nodes_list = graph.keys()

        nodes_for_graph = []
        id_num = 0

        for node in nodes_list:
            new_node = Node(None, None, node.long_lat, None, id_num)
            connections = {}
            nodes_for_graph.append(new_node)
            id_num += 1

        with open("nodes_for_graph_1.pkl", "wb") as fp:
            pickle.dump(nodes_for_graph, fp)

        with open("graph_1.pkl", "wb") as fp:
            pickle.dump(graph, fp)

    for graph_node in nodes_for_graph:
        graph_node.convert_connections_for_graph()
        for node in nodes_list:
            if node.long_lat == graph_node.long_lat:
                for connection in graph[node]:
                    for other_graph_node in nodes_for_graph:
                        if connection[0].long_lat == other_graph_node.long_lat:
                            graph_node.connections[other_graph_node] = connection[1]

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
