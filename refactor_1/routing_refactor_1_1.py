import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import copy
from operator import attrgetter
import time

"""# Get script and dataset file paths.
SCRIPT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_PATH, "DummyData.shp")
# Read the .shp file via geopanadas and store as a pandas dataframe.
ROUTES = gpd.read_file(DATA_PATH)
ROUTE_DATA = pd.DataFrame(ROUTES)
GRAPH = False
"""

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


# TODO: Incorporate the house task importance into the heuristic function.
def heuristic(start_node, end_node):
    """This function returns the shortest euclidean distance between two nodes. NOTE: This currently only works for
    2D data."""

    # TODO: Update the current function to incorporate a third dimension (elevation) when attempting to find the
    #  shortest euclidean distance between two nodes.

    start_node_x = start_node.lat
    start_node_y = start_node.longt

    end_node_x = end_node.lat
    end_node_y = end_node.longt

    diff_x = (start_node_x - end_node_x) ** 2
    diff_y = (start_node_y - end_node_y) ** 2

    shortest_possible_distance = np.sqrt(diff_x + diff_y)

    return shortest_possible_distance


# TODO: Make this function more efficient.
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


# TODO: FIX THE FUNCTION
"""def old_all_search():
    def search_round_routes(start_node):
        best_path = 10000
        nodes_been = [start_node]
        nodes_to_go = [node for node in GRAPH.nodes]
        indicies = []

        index = 0
        curr_time = 0

        curr_node = start_node

        while len(nodes_been) != len(GRAPH.nodes):
            indicies.append(index)
            hPath = search_next_node(nodes_to_go, curr_time, nodes_been, curr_node, index)
            if hPath == 0 or hPath == - 1:
                nodes_to_go.append(curr_node)
                indicies.remove(index)
                index = indicies[-1] + 1

            elif hPath + curr_time > best_path:
                print("best_path is better than all other paths!")
                nodes_been.remove(curr_node)
                nodes_to_go.append(curr_node)
                indicies.remove(index)
                index = indicies[-1] + 1

            else:
                node = curr_node.nodes[index]
                nodes_been.append(curr_node)
                if node in nodes_to_go:
                    nodes_to_go.remove(node)
                curr_node = node
                index = 0

        return nodes_been

    def search_next_node(nodes_to_go, curr_time, nodes_been, curr_node, index):
        if len(nodes_to_go) == 0:
            return 0
        elif len(curr_node.connections) == 0 or len(curr_node.connections) <= index:
            return -1
        return heuristic(curr_node.nodes[index], curr_node) * 4146.282847732093 + curr_time"""


def all_node_search():
    """# branch and bound search implementation???

    # resource: https://www.techiedelight.com/travelling-salesman-problem-using-branch-and-bound/

    # Matrix below is a copy of the one in the article

    import math
    import numpy as np

    matrix = [[math.inf, 20, 30, 10, 11],
              [15, math.inf, 16, 4, 2],
              [3, 5, math.inf, 2, 4],
              [19, 6, 18, math.inf, 3],
              [16, 4, 7, 16, math.inf]]

    matrix = np.array(matrix)"""

    def row_reduction(matrix):
        # Get minimum values of all the rows
        min_values = np.min(matrix, axis=1)

        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                # Ideally, the min_values[row] would never be "Inf", so maybe we should remove that if check
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[row]
        print(matrix)
        return matrix, min_values

    def col_reduction(matrix):
        # Get minimum values of all the columns
        min_values = np.min(matrix, axis=0)

        for column in range(len(matrix[0])):
            for row in range(len(matrix)):
                # Ideally, the min_values[row] would never be "Inf", so maybe we should remove that if check
                if min_values[row] != math.inf and matrix[row][column] != math.inf:
                    matrix[row][column] -= min_values[column]
        print(matrix)
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
        matrix[node_from, :] = math.inf
        matrix[:, node_to] = math.inf
        matrix[node_to, node_from] = math.inf
        return matrix

    """def explore_edge_test(index_from, index_to, matrix):
        matrix[index_from, :] = math.inf
        matrix[:, index_to] = math.inf
        matrix[index_to, index_from] = math.inf
        return matrix"""

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

            """min = math.inf
            minNode = None
            for node in self.queue:
                if min  self.queue[node]:
                    min = self.queue[node]
                    minNode = node"""
            node = min(self.queue, key=attrgetter('cost'))
            # print(node.id, node.cost)
            del self.queue[node]
            return node

    def print_path(root):
        next_node = root
        print(next_node.id)
        next_node = next_node.traveling_salesman_path
        while (next_node != root):
            print(next_node.id)
            next_node = next_node.traveling_salesman_path

    def branch_and_bound(graph, start_node):
        cost_matrix = graph.convert_to_matrix()
        initial_cost, cost_matrix = calculate_cost(cost_matrix)
        pq = PriorityQueue()
        root = graph.nodes[start_node]
        closed_list = []

        """copy_copy = copy.deepcopy(cost_matrix)
        print(copy_copy)
        print("\n")
        print(explore_edge(root.id, graph.nodes[0].id, copy_copy))
        print("\n")
        copy_copy = copy.deepcopy(cost_matrix)
        print(explore_edge(root.id, graph.nodes[1].id, copy_copy))
        print("\n")
        copy_copy = copy.deepcopy(cost_matrix)
        print(explore_edge(root.id, graph.nodes[2].id, copy_copy))
        print("\n")
        copy_copy = copy.deepcopy(cost_matrix)
        print(explore_edge(root.id, graph.nodes[3].id, copy_copy))
        print("\n")
        
        copy_copy = copy.deepcopy(cost_matrix)
        print(explore_edge(root.id, graph.nodes[4].id, copy_copy))
        print("\n")
        copy_copy = copy.deepcopy(cost_matrix)"""
        pq.push(root, 0)

        while not pq.isEmpty():
            """for node in pq.queue:
                print(node.id, pq.queue[node])"""

            parent = pq.pop()
            # print(parent.id, parent.cost)

            # print(closed_list, "length: ", len(closed_list))
            # print(len(graph.nodes))

            if closed_list == len(graph.nodes) - 1:  # This never gets called. :(
                parent.traveling_salesman_path = root
                print_path(root)
                return

            # lower_bound, cost_matrix = calculate_cost(cost_matrix)
            # print(cost_matrix)
            # print(lower_bound)
            cost_matrix_copy = copy.deepcopy(cost_matrix)

            print(parent.id)
            for sub_node in parent.connections:
                total_cost = initial_cost
                # cost_matrix_copy = copy.deepcopy(cost_matrix)
                if sub_node not in closed_list:
                    print(cost_matrix_copy[sub_node.id, parent.id])
                    total_cost += cost_matrix[sub_node.id, parent.id]
                    # print("   Node", sub_node.id, "Cost", total_cost)
                    # print("\n")

                    cost_matrix_copy = explore_edge(parent.id, sub_node.id, cost_matrix_copy)
                    cost_for_step, cost_matrix_copy = calculate_cost(cost_matrix_copy)
                    total_cost += cost_for_step
                    sub_node.cost = total_cost
                    pq.push(sub_node, sub_node.cost)
                print("    ", sub_node.id, sub_node.cost)
            closed_list.append(parent)

    """price, matrix = calculate_cost(matrix)
    matrix = explore_edge_test(0, 2, matrix)
    price, matrix = calculate_cost(matrix)
    print(matrix)"""
    branch_and_bound(GRAPH, 0)

    """price, matrix = calculate_cost(matrix)
    print("\nMatrix:")
    print(matrix, "\n")
    print("Cost: ", price)"""

    # matrix = GRAPH.convert_to_matrix()
    # print(matrix)

    """
    for row in range(len(matrix)):
      for col in range(len(matrix[0])):
        print(col, end = " ")
      print("")
    """


def testing(connection_testing, time_dist):
    """Function to contain all of the testing functions. This is just to reduce space."""

    def connections():
        """Lists all of the nodes and the bidirectional connections between them and other nodes."""

        # From the connection_testing parameter, retrieve the graph data.
        graph = connection_testing[1]
        print("LISTING CONNECTIONS PRESENT IN THE DATA...\n")

        # Iterate through all of the nodes, and for every node, iterate through the connections. For each of the
        # node-node connections, print the id, long, lati, and time for the child node being connected to from the
        # parent node.

        for node in graph.nodes:
            print(node.id, node.lat, node.longt)
            for connect in node.connections:
                print("    Connection to Node %s: " % connect.id, connect.lat, connect.longt, node.connections[connect])

    def time_per_long_lat():
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

    # Check which testing functions that we should be running.
    if connection_testing[0]:
        connections()
    if time_dist[0]:
        time_per_long_lat()


def main():
    global GRAPH

    start_time = time.time()
    # processed_data = preprocessing(ROUTE_DATA)

    processed_data = [[20, [0, 0], [1,1], 0],
                      [30, [0, 0], [2,2], 0],
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
    processed_data = pd.DataFrame(processed_data)
    processed_data.columns = ["Time", "Starting coordinates", "Finishing coordinates", "Importance"]
    print(processed_data)
    print("\n")

    GRAPH = Graph(processed_data)
    print(GRAPH.convert_to_matrix())
    """for node in GRAPH.nodes:
        print(node.id, node.lat, node.longt)
        for connections in node.connections:
            print("    Node:", connections.id)
            print("          Long/Lat:", connections.lat, connections.longt)
            print("          Time:", node.connections[connections])
    print("\n")"""

    direct_route = node_to_node_search(GRAPH.nodes[5], GRAPH.nodes[1])
    # hit_all_route = search_round_routes(GRAPH.nodes())

    testing(connection_testing=[False, GRAPH],
            time_dist=[False, GRAPH])

    print("\nMinutes since execution:", (time.time() - start_time) / 60)


if __name__ == "__main__":
    main()
