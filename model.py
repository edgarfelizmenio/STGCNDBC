import os
import dgl
import torch

os.environ['DGLBACKEND'] = "pytorch" 
import dgl.function as fn
import torch.nn as nn

# Graph Data
    # g.ndata['x'] = torch.Tensor(adata.X.todense())
    # g.ndata['init_coords'] = g.ndata['coords'] = torch.Tensor(coords)
    # g.edata['d'] = torch.Tensor(distances)
    # g.edata['w'] = torch.Tensor(w)

def weighted_mean_reduction(nodes):
    no

def calculate_similarity(edges):
    print(edges.src)
    print(edges.dst)

class SCConv(nn.Module):
    def __init__(self):
        super(SCConv, self).__init__()

    def forward(self, g, h):
        """Forward computation
        
        Parameters:
        -----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            # update nodes
            # update edges
            pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = SCConv()

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h

def train(g, model):
    features = g.ndata['x']
    g.ndata['embedding'] = g.ndata['x']
    for e in range(200):
        model(g, features)

print(calculate_similarity(g.edges))