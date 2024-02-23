import os
import dgl
import ot
import torch

os.environ['DGLBACKEND'] = "pytorch"
import numpy as np
import torch.nn.functional as F

rng = np.random.default_rng()

def create_dgl_graph(adata):
    n_nodes = adata.n_obs
    coords = adata.obsm['spatial']

    u = [(i,) * (n_nodes - (i + 1)) for i in range(n_nodes)]
    u = tuple(item for sublist in u for item in sublist)
    v = tuple(j for i in range(n_nodes) for j in range(i + 1, n_nodes))

    g = dgl.graph((u,v), num_nodes=n_nodes)
    g.ndata['x'] = torch.Tensor(adata.X.todense())
    g.ndata['init_coords'] = g.ndata['coords'] = torch.Tensor(coords)
    g = dgl.add_reverse_edges(g, copy_edata=True)

    # distance: smaller - closer, bigger - farther
    distances = ot.dist(g.ndata['init_coords'], metric='euclidean')
    
    # weights: 0 - closer, 1 - farther
    weights = F.normalize(distances, p=2)

    g.edata['d'] = distances[g.edges()]
    g.edata['w'] = weights[g.edges()]
    return g

def create_random_dgl_subgraph(g, size):
    subgraph_nodes = rng.choice(g.num_nodes(), size, replace=False)
    subgraph = g.subgraph(subgraph_nodes)
    return subgraph

def create_networkx_graph():
    pass