import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time

# Get script and dataset file paths.
SCRIPT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_PATH, "DummyData.shp")
# Read the .shp file via geopanadas and store as a pandas dataframe.
ROUTES = gpd.read_file(DATA_PATH)
ROUTE_DATA = pd.DataFrame(ROUTES)
GRAPH = False


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
        time = processed_raw_data.iloc[rows, 1]
        coordinates = list(processed_raw_data.iloc[rows, 2].coords)  # TODO: Ask about the .coords method
        start_longitude = round(coordinates[0][0], 3)
        start_latitude = round(coordinates[0][1], 3)

        end_longitude = round(coordinates[-1][0], 3)
        end_latitude = round(coordinates[-1][1], 3)
        processed_data.append([time, [start_longitude, start_latitude], [end_longitude, end_latitude]])

    processed_data = pd.DataFrame(processed_data)
    processed_data = processed_data.rename(columns={0: "Time", 1: "Starting coordinates", 2: "Finishing coordinates"})

    return processed_data


# TODO: Comment through the classes, I don't have the brains for this, so Rachel pop off bro.
class Node:
    def __init__(self, lat, longt, id):
        self.lat = lat
        self.longt = longt
        self.connections = {}
        self.nodes = []
        self.times = []
        self.id = id
        self.f = None
        self.g = None
        self.h = None

    def add_connection(self, node, time):
        for key in self.connections:
            if key == node:
                return -1
        self.connections[node] = time
        self.nodes.append(node)
        self.times.append(time)
        return 0


class Graph:
    def __init__(self, dataframe):
        self.nodes = []
        for rows in range(len(dataframe)):
            time = dataframe.loc[rows, "Time"]
            startCoor = dataframe.loc[rows, "Starting coordinates"]
            endCoor = dataframe.loc[rows, "Finishing coordinates"]
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
                startNode = Node(startCoor[0], startCoor[1], len(self.nodes))
                self.nodes.append(startNode)

            if endNode is None:
                endNode = Node(endCoor[0], startCoor[1], len(self.nodes))
                self.nodes.append(endNode)

            startNode.add_connection(endNode, time)
            endNode.add_connection(startNode, time)


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


# TODO: FIX THE NEXT THREE FUNCTIONS
def node_to_node_search(start_node, goal):
    from operator import attrgetter

    checking_list = [start_node]
    checked_list = []
    final_route = []

    start_node.f = 0
    start_node.g = 0
    tree_dict = {start_node: 0}

    level = 0
    while len(checking_list) != 0:
        current_node = checking_list[0]

        # Set the new node to be the previous sub_node with the lowest f(x).
        if current_node != start_node:
            current_node = min(checking_list, key=attrgetter('f'))

        #
        level = tree_dict[current_node]
        checking_list.remove(current_node)

        if level == len(final_route):
            final_route.append(current_node)
        else:
            final_route[level] = current_node

        # Add the current node to the checked list
        checked_list.append(current_node)

        # Check if the goal node has been reached
        if current_node == goal:
            return final_route

        # For the current_node, go through all of its connections, and determine their f(x) as a sum of their h(x)
        # and g(x).
        for sub_node in current_node.connections:
            if sub_node in checked_list:
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

            checking_list.append(sub_node)

        print("Current Node:", current_node.id, current_node.f)
        for sub_node in checking_list:
            print("    Sub-Nodes Data:", sub_node.id, sub_node.f, len(checking_list))
        print("\n")


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
    return heuristic(curr_node.nodes[index], curr_node) * 4146.282847732093 + curr_time


def testing(connection_testing=list, time_dist=list):
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
        time = 0
        # Iterate through all of the nodes, and for every node, iterate through the connections. For each of the
        # start-end node pairs, summate all of the collective distances and times.
        for node in graph.nodes:
            for connection in node.connections:
                dist += heuristic(node, connection)
                time += node.connections[connection]
                count += 1

        print("\nAverage Time Per Distance: ", time / dist)

    # Check which testing functions that we should be running.
    if connection_testing[0]:
        connections()
    if time_dist[0]:
        time_per_long_lat()


def main():
    global GRAPH

    start_time = time.time()
    processed_data = preprocessing(ROUTE_DATA)
    GRAPH = Graph(processed_data)

    direct_route = node_to_node_search(GRAPH.nodes[5], GRAPH.nodes[1])
    hit_all_route = search_round_routes(GRAPH.nodes())

    testing(connection_testing=[False, GRAPH],
            time_dist=[False, GRAPH])

    print("\nMinutes since execution:", (time.time() - start_time) / 60)


if __name__ == "__main__":
    main()
