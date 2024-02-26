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
def send_neighborhood_data(edges):
    # print("feats:", edges.src['x'].shape)
    # print("edge weights:", (1 - edges.data['w']).view(edges.batch_size(),1).shape)
    # adj_x = 
    # print("adj_x:", adj_x)
    # print("adj_x:", adj_x.shape)
    return {
        'adj_x': edges.src['x'] * (1 - edges.data['w']).view(edges.batch_size(),1),
        'adj_w': (1 - edges.data['w'])
        }

# user-defined reduction function for updating graph nodes
def reduce_neighborhood_score(nodes):
    # print(nodes.batch_size())
    # print("data shape:", nodes.data['x'].shape)    
    # print("adj_x: ", nodes.mailbox['adj_x'])
    # print("adj_x size: ", nodes.mailbox['adj_x'].shape)
    # print("adj_x_sum: ", nodes.mailbox['adj_x'].sum(1))
    # print("adj_x_sum_size: ", nodes.mailbox['adj_x'].sum(1).shape)
    # print("numerator:", nodes.data['x'] + nodes.mailbox['adj_x'].sum(1))
    # print("numerator shape:", (nodes.data['x'] + nodes.mailbox['adj_x'].sum(1)).shape)
    # print("data shape:", nodes.data['x'].shape)    
    # print("adj_w size: ", nodes.mailbox['adj_w'].sum(1).shape)
    # print("adj_w size: ", nodes.mailbox['adj_w'].sum(1).shape)    
    # print("1p_adj_w size: ", (1 + nodes.mailbox['adj_w'].sum(1)).shape)
    # print('adj_w:', nodes.mailbox['adj_w'])
    # print('adj_w shape:', nodes.mailbox['adj_w'].shape)
    # print('adj_w sum:', nodes.mailbox['adj_w'].sum(1))
    # print('adj_w sum:', nodes.mailbox['adj_w'].sum(1).view(nodes.batch_size()).shape)
    # print("denominator:", (1 + ))
    numerator = (nodes.data['x'] + nodes.mailbox['adj_x'].sum(1))
    denominator = (1 + nodes.mailbox['adj_w'].sum(1)).view(nodes.batch_size(),1)
    # print('denominator.shape:', denominator.shape)
    w_neighborhood = numerator/denominator
    # similarity = ot.dist

    
    return {'x': w_neighborhood}

# user-defined function for updating graph edges
def update_edge_weights(edges):
    # 1 if vectors are similar, 0 if not.
    # print(edges.src['x'].shape)
    # print(edges.dst['x'].shape)
    # print(edges.src['x'])
    # print(edges.src['x'].shape)
    # print(edges.dst['x'])
    # print(edges.dst['x'].shape)
    cos_sim = torch.nn.functional.cosine_similarity(edges.src['x'], edges.dst['x'], dim=1)
    cos_sim = (cos_sim + 1)/2
    cos_dist = 1 - cos_sim

    # print("cos_dist:", cos_dist)
    # print("cos_dist shape:", cos_dist.shape)

    # print('w:', edges.data['w'])
    # print('w shape:', edges.data['w'].shape)    
    new_weights = (edges.data['w'] + cos_dist)/(1 + cos_dist)
    # print('new_weights:', new_weights)
    # print('new weights shape:', new_weights.shape)
    # print(new_weights == edges.data['w'])
    # print(new_weights == edges.data['w'])
    # print(new_weights.shape)
    # new_weights = F.normalize(new_weights,dim=0)
    # print(new_weights)
    return {'w': new_weights}

class SCConv(nn.Module):
    def __init__(self):
        super(SCConv, self).__init__()

    def forward(self, g, features):
        """Forward computation
        
        Parameters:
        -----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['x'] = features['x']
            g.edata['w'] = features['w']
            g.update_all(message_func=send_neighborhood_data, reduce_func=reduce_neighborhood_score)
            g.apply_edges(update_edge_weights)
            # # send messages
            # send_neighborhood_data(g.edges())
            # # update nodes
            # reduce_neighborhood_score(g.nodes())
            # # update edges
            # update_edge_weights(g.edges())
            return {'x': g.ndata['x'], 
                    'w': g.edata['w']}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = SCConv()

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h
    
def train(g, model):
    x = g.ndata['x']
    
    g.ndata['embedding'] = g.ndata['x']
    for e in range(200):
        model(g, features)