"""
The following functions work perfectly, don't doubt them whatsoever:


"""



import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import copy
from operator import attrgetter
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


class PriorityQueue():
    def __init__(self):
        self.queue = {}

    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, node, priority):
        self.queue[node] = priority

    def inQueue(self, node):
        for node in self.queue:
            if node.id == node.id:
                return True
        return False

    def pop(self):
        node = min(self.queue, key=attrgetter('cost'))
        cost = self.queue[node]
        # print(node.id, node.cost)
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


def explore_edge(node_from, node_to, matrix):
    # Set rows to math.inf.
    matrix[node_from, :] = math.inf

    # Set columns to math.inf.
    matrix[:, node_to] = math.inf

    # Set (i,j) and (j,i) to math.inf.
    matrix[node_to, node_from] = math.inf

    return matrix


def print_path(root):
    next_node = root
    print(next_node.id)
    next_node = next_node.traveling_salesman_path
    while next_node != root:
        print(next_node.id)
        next_node = next_node.traveling_salesman_path


def branch_and_bound(start_node, original_matrix, graph):
    initial_cost, cost_matrix = calculate_cost(original_matrix)
    # print(cost_matrix)
    priority_queue = PriorityQueue()
    parent_node = graph.nodes[start_node]
    closed_list = []

    priority_queue.push(parent_node, initial_cost)

    while not priority_queue.isEmpty():
        parent, cost = priority_queue.pop()

        """if closed_list == len(graph.nodes) - 1:  # This never gets called. :(
            parent.traveling_salesman_path = parent_node
            print_path(parent_node)
            return"""

        # testing(matrix=cost_matrix_copy)

        print(parent.id, cost)

        for sub_node in parent.connections:
            print(sub_node.id)
            cost_matrix_copy = copy.deepcopy(cost_matrix)
            """for item in closed_list:
                print("   ", sub_node.id,  item.id, item.cost)"""
            total_cost = cost
            print("Node Cost: ", total_cost)
            if sub_node not in closed_list:
                total_cost += cost_matrix[parent.id, sub_node.id]
                print("Edge Cost: ", cost_matrix[parent.id, sub_node.id])

                cost_matrix_copy = explore_edge(parent.id, sub_node.id, cost_matrix_copy)

                cost_for_step, cost_matrix_copy = calculate_cost(cost_matrix_copy)
                print("LB Cost: ", cost_for_step)

                if sub_node.id == 4 or sub_node.id == 2:
                    testing(matrix = cost_matrix_copy)

                total_cost += cost_for_step
                print("Total Cost: ", total_cost, "\n")
                sub_node.cost = total_cost
                priority_queue.push(sub_node, sub_node.cost)
            # print("    ", sub_node.id, sub_node.cost)
        print("------------------------")
        closed_list.append(parent)


def testing(data_frame=None, graph=None, matrix=None):
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

    # Check what to print.
    if data_frame is not None:
        view_data()
    if graph is not None:
        view_connections()
    if matrix is not None:
        view_cost_matrix()


def sample_data(sample_matrix_number):
    """Returns one of the following matrices to be used throughout the code for testing."""

    # Get the sample matrix of choice.
    if sample_matrix_number == 1:
        # Triangle + Dead end.
        testing_data = [[10, [0, 0], [1, 1], 0],
                        [10, [1, 1], [2, 2], 0],
                        [60, [1, 1], [3, 3], 0],
                        [800, [0, 0], [3, 3], 0]]

    elif sample_matrix_number == 2:
        # Diamond + Dead end.
        testing_data = [[10, [0, 0], [1, 1], 0],
                        [10, [1, 1], [2, 2], 0],
                        [20, [2, 2], [3, 3], 0],
                        [300, [2, 2], [4, 4], 0],
                        [500, [0, 0], [4, 4], 0]]

    elif sample_matrix_number == 3:
        # Based on the tutorial article.
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

    else:
        sample_matrix_number = [[10, [0, 0], [1, 1], 0],
                                [10, [1, 1], [2, 2], 0],
                                [60, [1, 1], [3, 3], 0],
                                [800, [0, 0], [3, 3], 0]]

    # Turn the chosen matrix into a pd.DataFrame, and set the column labels.
    testing_data = pd.DataFrame(testing_data)
    testing_data.columns = ["Time", "Starting coordinates", "Finishing coordinates", "Importance"]

    return testing_data


def main():
    # If we want to test the code using our sample matrices.
    if TESTING:
        sample_number = 3
        data = sample_data(sample_number)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

        if sample_number == 3:
            matrix_data = [[math.inf, 20, 30, 10, 11],
                           [15, math.inf, 16, 4, 2],
                           [3, 5, math.inf, 2, 4],
                           [19, 6, 18, math.inf, 3],
                           [16, 4, 7, 16, math.inf]]
            matrix_data = np.array(matrix_data)
            print("Path should be: 0 3 1 4 2\n")


    # If we are using the dummy data set:
    else:
        data = preprocessing(ROUTE_DATA)

        graphical_data = Graph(data)
        matrix_data = graphical_data.convert_to_matrix()

    # testing(matrix=matrix_data)
    branch_and_bound(0, matrix_data, graphical_data)

    print("\nMinutes since execution:", (time.time() - START_TIME) / 60)


if __name__ == "__main__":
    main()
