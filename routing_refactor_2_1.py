"""
The following functions work perfectly, don't doubt them whatsoever:
    row_reduction
    col_reduction
    calculate_cost
    explore_edges
    ????
    ????
    testing
    sample_data
    main
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import copy
from operator import attrgetter
import matplotlib.pyplot as plt
import networkx as nx
import time

# Get script and dataset file paths.
SCRIPT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_PATH, "fake_dummy_data/DummyData.shp")
# Read the .shp file via geopanadas and store as a pandas dataframe.
ROUTES = gpd.read_file(DATA_PATH)
ROUTE_DATA = pd.DataFrame(ROUTES)

# Get the start time.
START_TIME = time.time()


# TODO: Incorporate multi-vist modifications for the importance factor
def preprocessing(unprocessed_raw_data):
    """Clean the .shp file that contains the route data. Create a second pandas data frame to store a processed
    version of the original data from the .shp file. """

    # Drop one of the two columns containing data id numbers, and drop the rows with any column containing NaN data.
    processed_raw_data = unprocessed_raw_data.drop("id", axis=1).dropna()
    # Reset the modified data frame index number
    processed_raw_data = processed_raw_data.reset_index()

    # Create a secondary pandas data frame that contains the index of nodes and the start/end longitude and latitude.
    processed_data = []

    # TODO: Maybe there's a more efficient way to do this than to loop through the entire unprocessed data set
    for rows in range(len(processed_raw_data.index)):
        route_time = processed_raw_data.iloc[rows, 1]
        coordinates = list(processed_raw_data.iloc[rows, 2].coords)  # TODO: Ask about the .coords method
        start_longitude = round(coordinates[0][0], 3)
        start_latitude = round(coordinates[0][1], 3)

        end_longitude = round(coordinates[-1][0], 3)
        end_latitude = round(coordinates[-1][1], 3)
        processed_data.append([route_time, [start_longitude, start_latitude], [end_longitude, end_latitude]])

    processed_data = pd.DataFrame(processed_data)
    processed_data = processed_data.rename(columns={0: "Time", 1: "Starting coordinates", 2: "Finishing coordinates"})

    # Add the importance of each house to the processed_data
    processed_data["Importance"] = [4, 10, 2, 5, 2, 6, 3, 6, 4, 2, 2, 5, 5, 5, 4, 7, 5, 5, 1, 5, 1, 7, 2, 6, 2, 2, 4, 3,
                                    1, 8, 2, 2]

    return processed_data


# TODO: Comment through the classes, I don't have the brains for this, so Rachel pop off bro.
class Node:
    def __init__(self, lat, longt, id, importance):
        self.lat = lat
        self.longt = longt
        self.connections = {}
        self.nodes = []
        self.times = []
        self.id = id
        self.f = None
        self.g = None
        self.h = None
        self.importance = importance
        self.traveling_salesman_path = None
        self.cost = 0

    def add_connection(self, node, route_time):
        for key in self.connections:
            if key == node:
                return -1
        self.connections[node] = route_time
        self.nodes.append(node)
        self.times.append(route_time)
        return 0


class Graph:
    def __init__(self, dataframe):
        self.nodes = []
        for rows in range(len(dataframe)):
            route_time = dataframe.loc[rows, "Time"]
            startCoor = dataframe.loc[rows, "Starting coordinates"]
            endCoor = dataframe.loc[rows, "Finishing coordinates"]
            importance = dataframe.loc[rows, "Importance"]
            startNode = None
            endNode = None
            for node in self.nodes:
                if (node.lat - startCoor[0] < 0.002 and startCoor[0] - node.lat < 0.002) and (
                        node.longt - startCoor[1] < 0.002 and startCoor[1] - node.longt < 0.002):
                    startNode = node
                if (node.lat - endCoor[0] < 0.002 and endCoor[0] - node.lat < 0.002) and (
                        node.longt - endCoor[1] < 0.002 and endCoor[1] - node.longt < 0.002):
                    endNode = node

            if startNode is None:
                startNode = Node(startCoor[0], startCoor[1], len(self.nodes), importance)
                self.nodes.append(startNode)

            if endNode is None:
                endNode = Node(endCoor[0], endCoor[1], len(self.nodes), importance)
                self.nodes.append(endNode)

            startNode.add_connection(endNode, route_time)
            endNode.add_connection(startNode, route_time)

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


class SolidStateNode():

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


class SolidStateTree():

    def __init__(self, root):
        self.root = root


# Following are the functions.
# TODO: Refactor the following code / make sure it works properly.
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

# TODO: Refactor the following code / make sure it works properly.
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
            return final_route

        # For the current_node, go through all of its connections, and determine their f(x) as a sum of their h(x)
        # and g(x).
        for sub_node in current_node.connections:
            if sub_node in closed_list:
                continue
            found = False
            for key, value in tree_dict.items():
                if key == sub_node:
                    found = True
            if not found:
                tree_dict[sub_node] = level + 1

            sub_node.g = current_node.connections[sub_node] + current_node.g
            sub_node.h = heuristic(sub_node, goal) * 4146.282847732093
            sub_node.f = sub_node.g + sub_node.h

            open_list.append(sub_node)

        print("Current Node:", current_node.id, current_node.f)
        for sub_node in open_list:
            print("    Sub-Nodes Data:", sub_node.id, sub_node.f, sub_node.importance)
        print("\n")


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


# TODO: Refactor the following code / make sure it works properly.
def print_path(root):
    next_node = root
    print(next_node.id)
    next_node = next_node.traveling_salesman_path
    while next_node != root:
        print(next_node.id)
        next_node = next_node.traveling_salesman_path


# TODO: Refactor the following code / make sure it works properly.
def search_solid_state_tree(root):
    if root.parent_node is not None:
        print("Parent: ", root.parent_node.node_id, "ID: ", root.node_id, "House: ", root.node.id)
    else:
        print("ID: ", root.node_id, "House: ", root.node.id)

    for child in root.children:
        search_solid_state_tree(child)


# TODO: Refactor the following code / make sure it works properly.
def tutorial_algorithm(start_node, original_matrix, graph):
    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []
    final_list = []
    visual_list = []

    priority_queue.push_with_matrix(parent_node, initial_cost, cost_matrix)

    while not priority_queue.isEmpty():
        parent, cost, parent_matrix = priority_queue.pop_with_matrix()

        if closed_list == len(graph.nodes) - 1:  # This never gets called. :(
            parent.traveling_salesman_path = parent_node
            # print_path(parent_node)
            return

        print(parent.id, cost)

        for sub_node in parent.connections:
            # print(sub_node.id)
            # Set initial cost to that of the parent cost.
            total_cost = cost
            parent_matrix_copy = copy.deepcopy(parent_matrix)

            if sub_node not in closed_list:
                # Add the cost of the edge to the total_cost. The edge cost must be retrieved from the starting node
                # reduced cost matrix (cost_matrix).
                edge_cost = cost_matrix[parent.id, sub_node.id]
                total_cost += edge_cost - sub_node.importance

                # Add the cost of the lower bound starting at the sub_node. This means that we need to add the
                # math.inf values, then perform a row and column reduction, and add the resulting cost to our total
                # cost. NOTE: This must be done on the current parent reduced matrix.
                parent_matrix_copy_explored = explore_edge(start_node, parent.id, sub_node.id, parent_matrix_copy)
                # print(parent_matrix_copy_explored)
                cost_for_step, parent_matrix_copy_explored_reduced = calculate_cost(parent_matrix_copy_explored)
                total_cost += cost_for_step

                sub_node.cost = total_cost

                priority_queue.push_with_matrix(sub_node, sub_node.cost, parent_matrix_copy_explored)

                # testing(costs_list=[cost, edge_cost, cost_for_step, total_cost])
                print("    ", sub_node.id, sub_node.cost)

        print("------------------------")
        closed_list.append(parent)
        visual_list.append([parent.id, cost])
        # print(final_list, visual_list)

    matrix_insert = 0
    for item in closed_list:
        if len(closed_list) - len(final_list) < len(closed_list):
            visual_list[matrix_insert].insert(1, item.id)
            matrix_insert += 1

        final_list.append(item.id)

        if len(closed_list) - len(final_list) == 0:
            visual_list[matrix_insert].insert(1, visual_list[matrix_insert][0])

    # print(final_list, visual_list)
    return final_list, visual_list


def g_g_algorithm(start_node, original_matrix, graph):
    return [], []


def prims_algorithm(start_node, original_matrix, graph):
    # Make a copy of the original matrix, this is needed for the Prims algorithm.
    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []
    final_list = []
    visual_list = []

    priority_queue.push_with_matrix(parent_node, initial_cost, cost_matrix)

    # A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
    # The program is for adjacency matrix representation of the graph

    class PrimsGraph():

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

            for cout in range(self.V):
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


def space_state_algorithm(start_node, original_matrix, graph):
    original_matrix_copy = copy.deepcopy(original_matrix)

    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix_copy)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []
    visual_list = []
    final_list = []

    root = Solid_State_Node(None, 0, parent_node, 0, cost_matrix, initial_cost)
    solid_state_tree = Solid_State_Tree(root)

    priority_queue.push_wo_matrix(root, initial_cost)

    i = 1
    num_of_backtracks = 0

    lowest_branch_cost = math.inf
    lowest_branch = []
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
            root_connection = False
            for connection in parent.connections:
                if connection.id == root.node.id:
                    root_connection = True

            if not root_connection:
                continue
                print("Can't find the root!")

            print("Found root...")

            parent_matrix_copy = copy.deepcopy(parent_matrix)

            parent_copy_explored = explore_edge(parent.id, parent.id, root.node.id, copy.deepcopy(parent_matrix_copy))
            cost_for_step, parent_copy_reduced = calculate_cost(parent_copy_explored)
            branch_cost = cost_for_step
            branch_cost += cost
            branch_cost += cost_matrix[parent.id, root.node.id]

            if not priority_queue.isEmpty():
                next_node, next_node_cost = priority_queue.peek()

                print(next_node_cost, branch_cost)

                if branch_cost <= next_node_cost:
                    while parent_state is not None:
                        return_val.append(parent_state.node.id)
                        parent_state = parent_state.parent_node

                    return_val.reverse()
                    return_val.append(0)
                    break

            if priority_queue.isEmpty():
                return_val = lowest_branch
            if lowest_branch_cost > branch_cost:
                lowest_branch_cost = branch_cost
                lowest_branch = []
                lowest_state_node = parent_state
                lowest_node = parent
                while lowest_state_node is not None:
                    lowest_branch.append(lowest_state_node.node.id)
                    print(lowest_state_node.node.id)
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
                # print(sub_node.id)
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
                # print(parent_matrix_copy_explored)
                cost_for_step, parent_matrix_copy_explored_reduced = calculate_cost(parent_matrix_copy_explored)
                total_cost += cost_for_step

                sub_node.cost = total_cost

                state_node = Solid_State_Node(parent_state, i, sub_node, parent_state.level + 1,
                                              parent_matrix_copy_explored, total_cost)
                i += 1

                # print(parent_state.level)

                priority_queue.push_wo_matrix(state_node, cost)

                # testing(costs_list=[cost, edge_cost, cost_for_step, total_cost])
                # print("    ", sub_node.id, sub_node.cost)
            # print(cost)
        # print("------------------------")
        closed_list.append(parent)
        # search_solid_state_tree(solid_state_tree.root)
        # print("Return val: ", return_val)

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


def testing(data_frame=None, graph=None, matrix=None, costs_list=None):
    """Function to contain all of the testing functions. This is just to reduce space."""

    def view_data():
        """Print the pandas DataFrame."""

        # Print to see the sample data in a pandas dataframe.
        print("SHOWCASING THE DATAFRAME...")
        print(data_frame)
        print("")

    def view_connections():
        """Lists all of the nodes and the bidirectional connections between them and other nodes."""

        # Iterate through all of the nodes, and for every node, iterate through the connections. For each of the
        # node-node connections, print the id, long, lati, and time for the child node being connected to from the
        # parent node.
        print("\nLISTING CONNECTIONS PRESENT IN THE DATA...")
        for node in graph.nodes:
            print(node.id, node.lat, node.longt)
            for connections in node.connections:
                print("    Node:", connections.id)
                print("         Long/Lat:", connections.lat, connections.longt)
                print("         Time:", node.connections[connections])
        print("")

    def view_cost_matrix():
        print(matrix)
        print("")

    def costs():
        print("Node Cost: ", costs_list[0])
        print("Edge Cost: ", costs_list[1])
        print("Lower Bound Cost: ", costs_list[2])
        print("Total Cost: ", costs_list[3])
        print()

    # Check what to print.
    if data_frame is not None:
        view_data()
    if graph is not None:
        view_connections()
    if matrix is not None:
        view_cost_matrix()
    if graph_data is not None:
        costs()


def sample_data(sample_matrix_number):
    """Returns one of the following matrices to be used throughout the code for testing."""

    # Get the sample matrix of choice.

    # Triangle + Dead end.
    if sample_matrix_number == 1:
        testing_data = [[10, [0, 0], [1, 1], 0],
                        [10, [1, 1], [2, 2], 0],
                        [60, [1, 1], [3, 3], 0],
                        [800, [0, 0], [3, 3], 0]]

        data_check = [0, 1, 2, 1, 3]

    # Diamond + Dead end.
    elif sample_matrix_number == 2:
        testing_data = [[10, [0, 0], [1, 1], 0],
                        [10, [1, 1], [2, 2], 0],
                        [20, [2, 2], [3, 3], 0],
                        [300, [2, 2], [4, 4], 0],
                        [500, [0, 0], [4, 4], 0]]

        data_check = [0, 1, 2, 3, 2, 4]

    # Based on the tutorial article.
    elif sample_matrix_number == 3:
        testing_data = [[20, [0, 0], [1, 1], 0],
                        [30, [0, 0], [2, 2], 0],
                        [10, [0, 0], [3, 3], 0],
                        [11, [0, 0], [4, 4], 0],
                        [15, [1, 1], [0, 0], 0],
                        [16, [1, 1], [2, 2], 0],
                        [4, [1, 1], [3, 3], 0],
                        [2, [1, 1], [4, 4], 0],
                        [3, [2, 2], [0, 0], 0],
                        [5, [2, 2], [1, 1], 0],
                        [2, [2, 2], [3, 3], 0],
                        [4, [2, 2], [4, 4], 0],
                        [19, [3, 3], [0, 0], 0],
                        [6, [3, 3], [1, 1], 0],
                        [18, [3, 3], [2, 2], 0],
                        [3, [3, 3], [4, 4], 0],
                        [16, [4, 4], [0, 0], 0],
                        [4, [4, 4], [1, 1], 0],
                        [7, [4, 4], [2, 2], 0],
                        [16, [4, 4], [3, 3], 0]]

        data_check = [0, 3, 1, 4, 2]

    # Geeks for Geeks TSP Example
    else:
        testing_data = [[10, [0, 0], [1, 1], 0],
                        [15, [0, 0], [2, 2], 0],
                        [20, [0, 0], [3, 3], 0],

                        [10, [1, 1], [0, 0], 0],
                        [35, [1, 1], [2, 2], 0],
                        [25, [1, 1], [3, 3], 0],

                        [15, [2, 2], [0, 0], 0],
                        [35, [2, 2], [1, 1], 0],
                        [30, [2, 2], [3, 3], 0],

                        [20, [3, 3], [0, 0], 0],
                        [25, [3, 3], [1, 1], 0],
                        [30, [3, 3], [2, 2], 0]]

        data_check = [0, 1, 3, 2]

    # Turn the chosen matrix into a pd.DataFrame, and set the column labels.
    testing_data = pd.DataFrame(testing_data)
    testing_data.columns = ["Time", "Starting coordinates", "Finishing coordinates", "Importance"]

    return testing_data, data_check


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

        nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
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

        nx.draw(H, pos, with_labels=True, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels_dict)
        # plt.savefig("simple_path.png")  # save as png

    plt.show()


def main():
    # General Parameters
    ALGORITHM = 4
    TESTING = False
    VISUALIZATION = True
    ideal_route = []

    # If we want to test the code using our sample matrices.
    if TESTING:
        sample_number = 4
        data, ideal_route = sample_data(sample_number)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

        # For sample 3, as the bidirectional weights are unequal, force the matrix to be the following
        if sample_number == 3:
            matrix_data = [[math.inf, 20, 30, 10, 11],
                           [15, math.inf, 16, 4, 2],
                           [3, 5, math.inf, 2, 4],
                           [19, 6, 18, math.inf, 3],
                           [16, 4, 7, 16, math.inf]]
            matrix_data = np.array(matrix_data)

    # If we are using the dummy data set, do the following.
    else:
        data = preprocessing(ROUTE_DATA)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

    # Determine the algorithmic route.
    if ALGORITHM == 1:
        final_path, visual_path = tutorial_algorithm(0, matrix_data, graphical_data)
    elif ALGORITHM == 2:
        final_path, visual_path = g_g_algorithm(0, matrix_data, graphical_data)
    elif ALGORITHM == 3:
        final_path, visual_path = prims_algorithm(0, matrix_data, graphical_data)
    else:
        final_path, visual_path = space_state_algorithm(0, matrix_data, graphical_data)

    if TESTING:
        if final_path == ideal_route:
            print("\nSuccess. The final path is equal to that of the brute force path.")
            print("    Desired Outcome: ", ideal_route)
            print("    Received Outcome: ", final_path)
        else:
            print("\nFailure. The final path is not equal to the of the brute force path.")
            print("    Desired Outcome: ", ideal_route)
            print("    Received Outcome: ", final_path)
    else:
        print("Received Outcome: ", final_path)

    print("\n------------------------")
    print("Minutes since execution:", (time.time() - START_TIME) / 60)

    if VISUALIZATION:
        visualization(graph_data=graphical_data, path=visual_path)


if __name__ == "__main__":
    main()
