import math

import igraph as ig
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def synapse_color(s):
    if "Apical" in s.tags:
        return "orange"
    if "Distal" in s.tags:
        return "purple"
    if "Proximal" in s.tags:
        return "green"
    return "black"


def visualise_network_structure(network):
    neuron_list = network.NeuronGroups
    synapse_list = network.SynapseGroups

    graph_nodes = list(range(len(neuron_list)))
    graph_edges = [(neuron_list.index(s.src), neuron_list.index(s.dst)) for s in synapse_list]

    graph = ig.Graph(len(neuron_list), graph_edges, directed=True)
    node_size = [math.sqrt(n.size) for n in neuron_list]
    max_size = max(node_size)
    for i in range(len(node_size)):
        node_size[i] = node_size[i] / max_size / 2
    graph.es["color"] = [synapse_color(s) for s in synapse_list]

    fig, ax = plt.subplots(figsize=(8, 8))

    ig.plot(graph,
            target=ax,
            layout="auto",
            vertex_size=node_size,
            vertex_color=["red" if (("inh" in n.tags) or ("GABA" in n.tags)) else "blue" for n in neuron_list],
            vertex_label=[n.tags[0] for n in neuron_list],
            vertex_label_size=7,
            edge_color=graph.es["color"])
    plt.show()


def plot_conv_weight(tensor):
    tensor = tensor.to('cpu')
    shape = tensor.size()

    fig, axs = plt.subplots(shape[0], shape[1], figsize=(shape[1] * 2 + 1, shape[2] * 2 + 1), dpi=100)
    cmap = cm.get_cmap('gray')
    normalizer = Normalize(vmin=0, vmax=1)
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i in range(shape[0]):
        for j in range(shape[1]):
            plt.subplot(shape[0], shape[1], i * shape[1] + j + 1).imshow(tensor[i][j], cmap=cmap, norm=normalizer)
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()
