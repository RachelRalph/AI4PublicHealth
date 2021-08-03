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
    def __init__(self, lat, longt, id, elevation):
        self.lat = lat
        self.longt = longt
        self.elevation = elevation
        self.connections = {}
        self.nodes = []
        self.times = []
        self.id = id
        self.f = None
        self.g = None
        self.h = None
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


# Following are the functions.
# TODO: Refactor the following code / make sure it works properly.
def heuristic(start_node, end_node):
    """This function returns the shortest euclidean distance between two nodes. NOTE: This currently only works for
    2D data."""

    # TODO: Update the current function to incorporate a third dimension (elevation) when attempting to find the
    #  shortest euclidean distance between two nodes.

    start_node_x = start_node.lat
    start_node_y = start_node.longt
    start_node_z = start_node.elevation

    end_node_x = end_node.lat
    end_node_y = end_node.longt
    end_node_z = start_node.elevation

    diff_x = (start_node_x - end_node_x) ** 2
    diff_y = (start_node_y - end_node_y) ** 2
    diff_z = (start_node_z - end_node_z) ** 2

    shortest_possible_distance = np.sqrt(diff_x + diff_y + diff_z)

    return shortest_possible_distance


def time_per_long_lat(connection_testing):
    """Based on the data, determine the amount of time needed to move one unit of latitude/longitude."""

    # From the connection_testing parameter, retrieve the graph data.
    graph = connection_testing[1]

    dist = 0
    count = 0
    route_time = 0
    # Iterate through all of the nodes, and for every node, iterate through the connections. For each of the
    # start-end node pairs, sum all of the collective distances and times.
    for node in graph.nodes:
        for connection in node.connections:
            dist += heuristic(node, connection)
            route_time += node.connections[connection]
            count += 1

    print("\nAverage Time Per Distance: ", route_time / dist)


# TODO: Refactor the following code / make sure it works properly.
def node_to_node_search(start_node, goal):
    open_list = [start_node]
    closed_list = []
    final_route = []

    start_node.f = 0
    start_node.g = 0
    tree_dict = {start_node.id: 0}

    while len(open_list) != 0:
        current_node = open_list[0]

        # Set the new node to be the previous sub_node with the lowest f(x).
        if current_node != start_node:
            current_node = min(open_list, key=attrgetter('f'))

        level = tree_dict[current_node.id]
        open_list.remove(current_node)

        if level == len(final_route):
            final_route.append(current_node)
        else:
            final_route[level] = current_node

        # Add the current node to the checked list
        closed_list.append(current_node)

        # Check if the goal node has been reached
        if current_node == goal:
            final_final_route = []

            for node in final_route:
                final_final_route.append(node.id)

            return final_final_route, final_final_route

        # For the current_node, go through all of its connections, and determine their f(x) as a sum of their h(x)
        # and g(x).
        for sub_node in current_node.connections:
            if sub_node in closed_list:
                continue
            found = False
            for key, value in tree_dict.items():
                print(sub_node.id, key, value)
                if key == sub_node:
                    found = True
            if not found:
                tree_dict[sub_node.id] = level + 1

            sub_node.g = current_node.connections[sub_node] + current_node.g
            sub_node.h = heuristic(sub_node, goal) * 4146.282847732093
            sub_node.f = sub_node.g + sub_node.h

            open_list.append(sub_node)

        print("Current Node:", current_node.id, level, current_node.f)
        for sub_node in open_list:
            print("    Sub-Nodes Data:", sub_node.id, sub_node.f, sub_node.importance)
        print("\n")



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
    if costs_list is not None:
        costs()


def sample_data(sample_matrix_number):
    """Returns one of the following matrices to be used throughout the code for testing."""

    # Get the sample matrix of choice.

    # Triangle + Dead end.
    if sample_matrix_number == 1:
        testing_data = [[10, [0, 0], [1, 1], 2147],
                        [10, [1, 1], [2, 2], 2000],
                        [60, [1, 1], [3, 3], 1896],
                        [800, [0, 0], [3, 3], 1849]]

        data_check = [0, 1, 2, 1, 3]

    # Diamond + Dead end.
    elif sample_matrix_number == 2:
        testing_data = [[10, [0, 0], [1, 1], 2342],
                        [10, [1, 1], [2, 2], 2123],
                        [20, [2, 2], [3, 3], 1823],
                        [300, [2, 2], [4, 4], 1946],
                        [500, [0, 0], [4, 4], 2312]]

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
    testing_data.columns = ["Time", "Starting coordinates", "Finishing coordinates", "Elevation"]

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
    TESTING = True
    VISUALIZATION = True
    ideal_route = []

    # If we want to test the code using our sample matrices.
    if TESTING:
        sample_number = 1
        data, ideal_route = sample_data(sample_number)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

        # For sample 3, as the bidirectional weights are unequal, force the matrix to be the following
        """if sample_number == 3:
            matrix_data = [[math.inf, 20, 30, 10, 11],
                           [15, math.inf, 16, 4, 2],
                           [3, 5, math.inf, 2, 4],
                           [19, 6, 18, math.inf, 3],
                           [16, 4, 7, 16, math.inf]]
            matrix_data = np.array(matrix_data)"""

    # If we are using the dummy data set, do the following.
    else:
        data = preprocessing(ROUTE_DATA)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

    # testing(graph=graphical_data, matrix = matrix_data)

    # Determine the algorithmic route.
    final_path, visual_path = node_to_node_search(graphical_data.nodes[0], graphical_data.nodes[3])

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

    """if VISUALIZATION:
        visualization(graph_data=graphical_data, path=visual_path)"""


if __name__ == "__main__":
    main()
