# Beverly named this file.

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

SCRIPT_PATH = os.path.dirname(__file__)

# Read the road line data .shp file via geopandas and store as a pandas dataframe.
ROAD_LINE_DATA = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_lines_CORRECT.shp")
ROAD_LINE_DATA = gpd.read_file(ROAD_LINE_DATA)
ROAD_LINE_DATA = pd.DataFrame(ROAD_LINE_DATA)

# Read the road points w/ elevation .shp file via geopandas and store as a pandas dataframe.
ROAD_POINT_WITH_ELEVATION_PATH = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_roads_pointdata_with_elevation.shp")
ROAD_POINT_WITH_ELEVATION_DATA = gpd.read_file(ROAD_POINT_WITH_ELEVATION_PATH)
ROAD_POINT_WITH_ELEVATION_DATA = pd.DataFrame(ROAD_POINT_WITH_ELEVATION_DATA)

# Read the building point data w/ elevation .shp file via geopandas and store as a pandas dataframe.
BUILDING_FILE_WITH_ELEVATION = os.path.join(SCRIPT_PATH, "Datasets/MZUZU_buildings_with_elevation.shp")
BUILDING_FILE_WITH_ELEVATION = gpd.read_file(BUILDING_FILE_WITH_ELEVATION)
BUILDING_FILE_WITH_ELEVATION = pd.DataFrame(BUILDING_FILE_WITH_ELEVATION)


# Get the start time.
START_TIME = time.time()


def road_line_processing(road_line_df):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
        version of the original data from the .shp file. """

    processed_data_line = []

    for rows in range(len(road_line_df.index)):
        coordinates_line = road_line_df.iloc[rows, 11]
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

    return processed_data_line


def line_node_dataframe(road_line_df):
    node_information = []

    for index, row in road_line_nodes.iterrows():
        for i in range(len(row["Coordinates List"])):
            connections = []
            coordinate_pair = row["Coordinates List"][i]

            if i != 0:
                connections.append(row["Coordinates List"][i - 1])

            if i != len(row["Coordinates List"]) - 1:
                connections.append(row["Coordinates List"][i + 1])
            node_information.append([coordinate_pair, connections])

    node_data = pd.DataFrame(node_data)

    return node_data


def node_dictionary(node_df):
    long_lat_pairs = {}

    multiple_index_pairs = []

    for index, row in node_df.iterrows():
        long_lat_coord = row[0]

        if long_lat_coord in long_lat_pairs.keys():
            long_lat_pairs[long_lat_coord].append(index)

            if long_lat_coord not in multiple_index_pairs:
                multiple_index_pairs.append(long_lat_coord)

        else:
            long_lat_pairs[long_lat_coord] = [index]


def intersection_node_check(node_df, node_dict, )

class Node:
    def __init__(self, road_condition, elevation, long_lat, connections, id):
        self.road_condition = road_condition
        self.elevation = elevation
        self.connections = connections
        self.long_lat = long_lat
        self.f = None
        self.g = None
        self.h = None
        self.id = id

    def convert_connections_for_graph(self, connections):
        self.connections = connections


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





node_data[2] = True

min_length = math.inf
for pair in multiple_index_pairs:
    index_list = lat_long_pairs[pair]

    if len(index_list) < min_length:
        min_length = len(index_list)

    first_index = index_list[0]

    for second_index in index_list[1:]:
        all_connections = list(set(node_data.at[first_index, 1] + node_data.at[second_index, 1]))
        node_data.at[first_index, 1] = all_connections
        node_data.at[second_index, 2] = False
        lat_long_pairs[pair] = [first_index]

node_data[3] = False

for index, row in node_data.iterrows():
    if len(row[1]) != 2:
        node_data.iat[index, 3] = True

for index, row in node_data.iterrows():
    if row[2]:
        if row[3]:
            continue
        else:
            if len(node_data.at[index, 1]) >= 2:
                connection1 = node_data.at[index, 1][0]
                connection2 = node_data.at[index, 1][1]
                index1 = lat_long_pairs[connection1][0]
                index2 = lat_long_pairs[connection2][0]
                node_data.at[index1, 1].remove(node_data.at[index, 0])
                node_data.at[index2, 1].remove(node_data.at[index, 0])
                node_data.at[index1, 1] = list(set(node_data.at[index1, 1] + [node_data.at[index2, 0]]))
                node_data.at[index2, 1] = list(set([node_data.at[index1, 0]] + node_data.at[index2, 1]))

node_data.reset_index().drop("index", axis=1)

node_data = node_data.reset_index()
node_data = node_data.drop("index", axis=1)
node_data = node_data.drop(2, axis=1)

node_data = node_data.rename(
    columns={0: "Long/Lat Coordinates", 1: "Connections", 3: "Is Intersection", })

node_dictionary = {}
for index, row in node_data.iterrows():
    if row["Is Intersection"]:
        row_node = Node(None, None, row["Long/Lat Coordinates"], row["Connections"])
        node_dictionary[row["Long/Lat Coordinates"]] = row_node

house_ids = []

random.seed(42)
for i in range(10):
    house_ids.append(random.randint(0, len(BUILDING_FILE_WITH_ELEVATION)))

nearest_intersections = []

for house_id in house_ids:
    print(house_id)
    coordinates = list(BUILDING_FILE_WITH_ELEVATION.loc[house_id, "geometry"].coords)
    long = coordinates[0][0]  # Contains max 7 decimal points
    lat = coordinates[0][1]  # Contains max 7 decimal points
    print(long, lat)
    elevation = BUILDING_FILE_WITH_ELEVATION.at[house_id, "SAMPLE_1"]
    near_intersection = None
    near_intersection_dist = math.inf
    for index, row in node_data.iterrows():
        if row["Is Intersection"]:
            node_long = row["Long/Lat Coordinates"][0]
            node_lat = row["Long/Lat Coordinates"][1]
            if near_intersection_dist > heuristic(lat, long, node_lat, node_long):
                near_intersection_dist = heuristic(lat, long, node_lat, node_long)
                near_intersection = node_dictionary[row["Long/Lat Coordinates"]]
    nearest_intersections.append(near_intersection)

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
        cost = node_to_node_search(closest_house, intersection)
        houses_completed.append(closest_house)
        houses_visited.append((closest_house, cost))
        graph[intersection] = houses_visited

nodes_list = graph.keys()

nodes_for_graph = []
id_num = 0

for node in nodes_list:
    new_node = Node(None, None, node.long_lat, None)
    new_node.id = id_num
    connections = {}
    for house_cost in graph[node]:
        print(house_cost[0], house_cost[1])
        connections[house_cost[0]] = house_cost[1]

    new_node.convert_connections_for_graph(connections)
    nodes_for_graph.append(new_node)
    id_num += 1

for key in nodes_for_graph[0].connections.keys():
    print(key.long_lat)

def heuristic(long1, lat1, long2, lat2):
    return math.sqrt((lat2 - lat1) ** 2 + (long2 - long1) ** 2)


def node_to_node_search(start_node, goal):
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
            if sub_node in node_dictionary.keys():
                sub_node = node_dictionary[sub_node]
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
        search_solid_state_tree(root)
        print("Hey! Checking priority queue...")

        parent_state, cost = priority_queue.pop_wo_matrix()
        # print(parent_state)

        parent = parent_state.node
        parent_matrix = parent_state.reduced_matrix

        # print("PARENT LEVEL: " , parent_state.level)
        if parent_state.level >= graph.size() - 1 + num_of_backtracks:
            if len(first_branch) == 0:
                first_branch_cost = parent_state.cost
                next_state = parent_state
                while next_state is not None:
                    first_branch.append(next_state.node.id)
                    next_state = next_state.parent_node
                first_branch.reverse()

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

                        return_val.reverse()
                        return_val.append(0)
                        break

                    else:
                        return_val = first_branch
                        break

            if priority_queue.isEmpty():
                if first_branch_cost * 2 < lowest_branch_cost:
                    print("Returning first branch...")
                    return_val = first_branch
                else:
                    return_val = lowest_branch

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

            print("Going to the top of the loop...")

            continue

        if cost > lowest_branch_cost:
            return_val = lowest_branch
            print("Found lowest cost")
            break
        print("Going on down...")

        if len(parent.connections) == 1:
            print("Backtracking...")
            num_of_backtracks = 1

        for sub_node in parent.connections:
            if sub_node not in parent_state.closedList or len(parent.connections) == 1:
                print(sub_node.id)
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
                visual_list[matrix_insert].insert(1, item.id)
                matrix_insert += 1

            final_list.append(item.id)

            if len(closed_list) - len(final_list) == 0:
                visual_list[matrix_insert].insert(1, visual_list[matrix_insert][0])

    return final_list, visual_list


def main():
    road_line_nodes = road_line_processing(BUILDING_FILE_WITH_ELEVATION)
    node_data = line_node_dataframe(road_line_nodes)
    node_dictionary(node_data)


if __name__ == "__main__":
    main()
