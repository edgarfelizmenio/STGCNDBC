import os
import dgl
import torch

os.environ['DGLBACKEND'] = "pytorch" 
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

# Graph Data
    # g.ndata['x'] = torch.Tensor(adata.X.todense())
    # g.ndata['init_coords'] = g.ndata['coords'] = torch.Tensor(coords)
    # g.edata['d'] = torch.Tensor(distances)
    # g.edata['w'] = torch.Tensor(w)

# def weighted_mean_reduction(nodes):
#     no

# user-defined function for sending graph nodes

# user-defined reduction function for updating graph nodes
def reduce_neighborhood_score(nodes):
    nodes['x'] += nodes.mailbox['x'] * nodes.mailbox['x']
    w_neighborhood = (1 - nodes.mailbox['incident_weights']) * nodes.mailbox['x']
    # multiply by 1-weight

    return {'wn': w_neighborhood}

# user-defined function for updating graph edges
def update_edge_weights(edges):
    new_weights = edges.data['w'] + edges.data['wu']
    print(new_weights.shape)
    new_weights = F.normalize(new_weights,dim=0)
    print(new_weights)
    return {'w': new_weights}

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