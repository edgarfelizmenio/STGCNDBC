import dgl

import networkx as nx
import numpy as np

def get_initial_dgl_graph_positions(graph):
    init_pos = graph.ndata['coords'].numpy()
    return {i: init_pos[i] for i in range(len(init_pos))}

def sample_networkx_from_dgl(graph):
    network = dgl.to_networkx(graph.cpu(), node_attrs=['x'], edge_attrs=['w'])
    return nx.DiGraph(network).to_undirected(reciprocal=True)

def get_network_weights(graph):
    network_weights = [w.item() for u,v,w in graph.edges.data('w')]
    minw, maxw = min(network_weights), max(network_weights)
    rangew = maxw - minw
    return network_weights, minw, maxw, rangew

def weighted_layout(G, weight_attr='w', init_pos=None):
    if init_pos is None:
        pos = nx.spring_layout(G, pos=init_pos)
    else:
        pos = init_pos 
    src = []
    dst = []
    data = []
    for u, v, e in G.edges(data=True):
        src.append(u)
        dst.append(v)
        data.append(e[weight_attr].numpy().item())

    minw, maxw = min(data), max(data)
    rangew = maxw - minw

    for u, v, w in zip(src,dst,data):
        force = ((1 - w) - minw)/rangew - 0.5
        vector = pos[v] - pos[u]
        angle = np.arctan2(vector[1], vector[0])
        pos[v][0] += force*np.cos(angle)
        pos[v][1] += force*np.sin(angle)

    return pos