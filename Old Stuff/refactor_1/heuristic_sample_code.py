import numpy as np
import matplotlib.pylab as plt

sample_data = {"O": [4, 7],
               "A": [2, 5],
               "B": [5, 3],
               "C": [5, 5],
               "D": [3, 10],
               "E": [6, 8]}

optimal_dist_list = {}


def h(data):
    global optimal_dist_list

    start_node = "O"
    for key in data.keys():
        start_node_x, start_node_y = data.get(start_node)
        end_node_x, end_node_y = data.get(key)

        diff_x = (start_node_x - end_node_x) ** 2
        diff_y = (start_node_y - end_node_y) ** 2

        optimal_distance = np.sqrt(diff_x + diff_y)
        optimal_dist_list[key] = optimal_distance


def showcase():
    data = sample_data
    x_vals_list = []
    y_vals_list = []
    labels = []
    for keys, vals in data.items():
        x_vals_list.append(vals[0])
        y_vals_list.append(vals[1])
        labels.append(keys)

    fig, ax = plt.subplots()
    ax.scatter(x_vals_list, y_vals_list)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_vals_list[i], y_vals_list[i]))

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    h(sample_data)
    print(optimal_dist_list)
    showcase()
