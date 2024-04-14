import igraph as ig
import matplotlib.pyplot as plt
import math
import torch
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def synapse_color(s):
    if "Apical" in s.tags:
        return "orange"
    if "Distal" in s.tags:
        return "purple"
    if "Proximal" in s.tags:
        return "green"
    return "black"


def visualize_network_structure(network):
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

    fig, ax = plt.subplots(figsize=(8,8))

    ig.plot(graph,
            target=ax,
            layout="auto",
            vertex_size=node_size,
            vertex_color=["red" if (("inh" in n.tags) or ("GABA" in n.tags)) else "blue" for n in neuron_list],
            vertex_label=[n.tags[0] for n in neuron_list],
            vertex_label_size=7,
            edge_color=graph.es["color"])
    plt.show()
