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
DATA_PATH = os.path.join(SCRIPT_PATH, "DummyData.shp")
# Read the .shp file via geopanadas and store as a pandas dataframe.
ROUTES = gpd.read_file(DATA_PATH)
ROUTE_DATA = pd.DataFrame(ROUTES)

# General Parameters
TESTING = True

# Get the start time.
START_TIME = time.time()


# TODO: Incorporate multi vist modifications for the importance factor
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


class PriorityQueue():
    def __init__(self):
        self.queue = {}

    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, node, priority):
        self.queue[node] = [priority]

    def inQueue(self, node):
        for node in self.queue:
            if node.id == node.id:
                return True
        return False

    def peek(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        return node, cost
        

    def pop(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        del self.queue[node]
        return node, cost


def row_reduction(matrix):
    # Get minimum values of all the rows
    min_values = np.min(matrix, axis=1)

    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            # Ideally, the min_values[row] would never be "Inf", so maybe we should remove that if check
            if min_values[row] != math.inf and matrix[row][column] != math.inf:
                matrix[row][column] -= min_values[row]

    return matrix, min_values


def col_reduction(matrix):
    # Get minimum values of all the columns
    min_values = np.min(matrix, axis=0)

    for column in range(len(matrix[0])):
        for row in range(len(matrix)):
            # Ideally, the min_values[row] would never be "Inf", so maybe we should remove that if check
            if min_values[row] != math.inf and matrix[row][column] != math.inf:
                matrix[row][column] -= min_values[column]

    return matrix, min_values


def calculate_cost(matrix):
    cost = 0

    matrix, min_row = row_reduction(matrix)
    matrix, min_col = col_reduction(matrix)

    for index in range(len(min_row)):
        if min_row[index] != math.inf:
            cost += min_row[index]
        if min_col[index] != math.inf:
            cost += min_col[index]

    return cost, matrix


def explore_edge(start_node, node_from, node_to, matrix):
    # Set rows to math.inf.
    matrix[node_from, :] = math.inf

    # Set columns to math.inf.
    matrix[:, node_to] = math.inf

    # Set (i,j) and (j,i) to math.inf.
    matrix[node_to, start_node] = math.inf

    return matrix


def print_path(root):
    next_node = root
    print(next_node.id)
    next_node = next_node.traveling_salesman_path
    while next_node != root:
        print(next_node.id)
        next_node = next_node.traveling_salesman_path


class Solid_State_Node():

    def __init__(self, parent_node, node_id, node, level, reduced_matrix, cost):
        self.node = node
        self.node_id = node_id
        self.reduced_matrix = reduced_matrix
        self.parent_node = parent_node
        self.level = level
        self.cost = cost
        self.children = []
        if parent_node != None:
            parent_node.add_children(self)
            self.closedList = parent_node.closedList + [parent_node.node]

        else:
            self.closedList = []

    def add_children(self, child):
        self.children.append(child)


class Solid_State_Tree():

    def __init__(self, root):
        self.root = root

def search_solid_state_tree(root):
    if root.parent_node != None:
        print("Parent: " , root.parent_node.node_id,"ID: ",  root.node_id,"House: ", root.node.id)
    else:
         print("ID: ",  root.node_id,"House: ", root.node.id)
        
    for child in root.children:
        search_solid_state_tree(child)


            
def branch_and_bound(start_node, original_matrix, graph):
    # Reduce the original matrix.
    initial_cost, cost_matrix = calculate_cost(original_matrix)

    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []
    final_list = []

    root = Solid_State_Node(None, 0, parent_node, 0, cost_matrix, initial_cost)
    solid_state_tree = Solid_State_Tree(root)

    priority_queue.push(root, initial_cost)

    i = 1
    num_of_backtracks = 0

    while not priority_queue.isEmpty():
        

        parent_state, cost = priority_queue.pop()

        parent = parent_state.node
        parent_matrix = parent_state.reduced_matrix

        print("PARENT LEVEL: " , parent_state.level)

        if parent_state.level >= graph.size() - 1 + num_of_backtracks:
            print(len(cost_matrix[0]))
            parent = parent_state

            if root in parent.connections:
                return_val = []
                calculate_cost 

            while parent != None:
                return_val.append(parent.node.id)
                parent = parent.parent_node

            return_val.reverse()
            return return_val

        print(parent.id, cost)

        sub_node_remove = False

        if len(parent.connections) == 1:
                sub_node_remove = True
                num_of_backtracks += 1


        for sub_node in parent.connections:
            if sub_node not in parent_state.closedList or len(parent.connections) == 1 or (len(parent.connections) == 2 and all(houses in parent.connections for houses in parent_state.closedList)):
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
                # print(parent_matrix_copy_explored)
                cost_for_step, parent_matrix_copy_explored_reduced = calculate_cost(parent_matrix_copy_explored)
                total_cost += cost_for_step

                sub_node.cost = total_cost

                state_node = Solid_State_Node(parent_state, i, sub_node, parent_state.level + 1, parent_matrix_copy_explored, total_cost)
                i += 1

                print(parent_state.level)

                priority_queue.push(state_node, cost)

                # testing(costs_list=[cost, edge_cost, cost_for_step, total_cost])
                # print("    ", sub_node.id, sub_node.cost)


        print("------------------------")
        closed_list.append(parent)
        search_solid_state_tree(solid_state_tree.root)

    
    for item in closed_list:
        final_list.append(item.id)
    return final_list






def testing(data_frame=None, graph=None, matrix=None, costs_list=[]):
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
    if len(costs_list) != 0:
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

def visualization(graph_data):
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

    nx.draw(G, pos, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict)
    # plt.savefig("simple_path.png")  # save as png
    plt.show()


def main():
    # If we want to test the code using our sample matrices.
    if TESTING:
        sample_number = 1
        data, data_checker = sample_data(sample_number)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

        if sample_number == 3:
            matrix_data = [[math.inf, 20, 30, 10, 11],
                           [15, math.inf, 16, 4, 2],
                           [3, 5, math.inf, 2, 4],
                           [19, 6, 18, math.inf, 3],
                           [16, 4, 7, 16, math.inf]]
            matrix_data = np.array(matrix_data)

    # If we are using the dummy data set:
    else:
        data = preprocessing(ROUTE_DATA)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

    # testing(matrix=matrix_data)
    testing(graph=graphical_data)
    # testing(data_frame=data, graph=graphical_data, matrix=matrix_data)
    final_path = branch_and_bound(0, matrix_data, graphical_data)

    if TESTING:
        if final_path == data_checker:
            print("\nSuccess. The final path is equal to that of the brute force path.")
            print("    Desired Outcome: ", data_checker)
            print("    Received Outcome: ", final_path)
        else:
            print("\nFailure. The final path is not equal to the of the brute force path.")
            print("    Desired Outcome: ", data_checker)
            print("    Received Outcome: ", final_path)
    else:
        print(final_path)

    print("\n------------------------")
    print("Minutes since execution:", (time.time() - START_TIME) / 60)

    visualization(graphical_data)

if __name__ == "__main__":
    main()
