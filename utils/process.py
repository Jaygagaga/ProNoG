import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_webkb(data, nb_nodes):
    nb_graphs = 1
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros((nb_nodes,5))
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    sizes = data.x.shape[0]
    features= data.x

    for i in range(data.y.shape[0]):
        labels[i][data.y[i]] = 1
    masks= 1.0
    e_ind = data.edge_index
    coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
    adjacency = coo.todense()
    adjacency = sp.csr_matrix(adjacency)
    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/home/xingtong/WebKB_node_origin/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/home/xingtong/WebKB_node_origin/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

# def sparse_to_tuple(sparse_mx, insert_batch=False):
#     """Convert sparse matrix to tuple representation."""
#     """Set insert_batch=True if you want to insert a batch dimension."""
#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):
#             mx = mx.tocoo()
#         if insert_batch:
#             coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
#             values = mx.data
#             shape = (1,) + mx.shape
#         else:
#             coords = np.vstack((mx.row, mx.col)).transpose()
#             values = mx.data
#             shape = mx.shape
#         return coords, values, shape
#
#     if isinstance(sparse_mx, list):
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)
#
#     return sparse_mx
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info




def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def process_tu(data, class_num):
    nb_nodes = data.num_nodes
    nb_graphs = data.num_graphs
    # print("len",nb_graphs)
    ft_size = data.num_features

    # print("data",data)
    # labels = np.zeros((nb_graphs, class_num))

    num = range(class_num)

    labelnum=range(class_num,ft_size)
    # features = np.zeros((nb_graphs, nb_nodes, ft_size))
    # adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    # labels = np.zeros(nb_graphs)
    # sizes = np.zeros(nb_graphs, dtype=np.int32)
    # masks = np.zeros((nb_graphs, nb_nodes))
    # zero = np.zeros((nb_nodes, nb_nodes))
    for g in range(nb_graphs):
        # print("g", g)
        if g == 0:
            # sizes = data[g].x.shape[0]
            features = data[g].x[:, num]
            # print("rawlabels[0]",data.y.shape)
            # print("rawlabels[0]",data.batch)
            # print("rawlabels[0]",data.batch[0])
            # print("rawlabels[0]",data.y[1])
            rawlabels = data[g].x[:, labelnum]
            # masks[g, :sizes[g]] = 1.0
            e_ind = data[g].edge_index
            # print("e_ind",e_ind)
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(features.shape[0], features.shape[0]))
            # print("coo",coo)
            adjacency = coo.todense()
        else:
            tmpfeature = data[g].x[:, num]
            features = np.row_stack((features, tmpfeature))
            tmplabel = data[g].x[:, labelnum]
            rawlabels = np.row_stack((rawlabels, tmplabel))
            e_ind = data[g].edge_index
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
            # print("coo",coo)
            tmpadj = coo.todense()
            zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
            tmpadj1 = np.column_stack((adjacency, zero))
            tmpadj2 = np.column_stack((zero.T, tmpadj))
            adjacency = np.row_stack((tmpadj1, tmpadj2))

    # for x in range(nb_graphs):
    #     if nb_graphs == 1:
    #         labels[0][rawlabels.item()] = 1
    #         break
    #     labels[x][rawlabels[x][0]] = 1

    # print("feature",features)
    # print("feature", features.size)
    # print("rawlabel",rawlabels)
    # print("rawlabel", rawlabels.size)


    nodelabels =rawlabels
    adj = sp.csr_matrix(adjacency)

    # graphlabels = labels

    return features, adj, nodelabels


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




#寻找当前节点的邻居节点
def find_neighbors(adj, node):
    neighbors = []
    for i in range(len(adj[node])):
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    return neighbors



#寻找当前节点的邻居节点和二阶邻居节点
# def find_2hop_neighbors(adj, node):
#     neighbors = []
#     # print(type(adj))
#     for i in range(len(adj[node])):
#         # print('i',i)
#         # print('node',node)
#         # print('adj[node][i]',adj[node,i])
#         if adj[node][i] != 0 and node != i:
#             neighbors.append(i)
#     neighbors_2hop = []
#     for i in neighbors:
#         for j in range(len(adj[i])):
#             if adj[i][j] != 0 and j != i:
#                 neighbors_2hop.append(j)
#     return neighbors, neighbors_2hop
import random

def find_2hop_neighbors(adj, node, k=20):
    neighbors = []
    # print(type(adj))

        # neighbor1hop_list = random.sample(list(range(0,len(adj[node]))), k)
    for i in range(len(adj[node])):
        if len(adj) >= 2000:
            if len(neighbors) >= k:
                break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        for j in range(len(adj[i])):
            if len(adj) >= 2000:
                if len(neighbors_2hop) >= k:
                    break
            if adj[i][j] != 0 and j != i and j != node:
                neighbors_2hop.append(j)
    return neighbors, neighbors_2hop


def find_3hop_neighbors(adj, node, k1=20,k2=30):
    neighbors = []
    # print(type(adj))

        # neighbor1hop_list = random.sample(list(range(0,len(adj[node]))), k)
    for i in range(len(adj[node])):
        if len(adj) >= 2000:
            if len(neighbors) >= k1:
                break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        for j in range(len(adj[i])):
            if len(adj) >= 2000:
                if len(neighbors_2hop) >= k1:
                    break
            if adj[i][j] != 0 and j != i and j != node:
                neighbors_2hop.append(j)
    neighbors_3hop = []
    for i in neighbors_2hop:
        for j in range(len(adj[i])):
            if len(adj) >= 2000:
                if len(neighbors_3hop) >= k2:
                    break
            if adj[i][j] != 0 and j != i and j != node:
                neighbors_3hop.append(j)
    return neighbors, neighbors_2hop, neighbors_3hop


import torch_geometric
def get_new_edge_index(subset_nodes,edge_index,count_nodes):
    new_edge_index, _ = torch_geometric.utils.subgraph(edge_index=edge_index, subset=subset_nodes,
                                                       relabel_nodes=True)
    dict = {i: i + count_nodes + 1 for i in np.unique(new_edge_index)}
    row = [dict[i]-1 for i in new_edge_index[0].tolist()]
    col = [dict[i]-1 for i in new_edge_index[1].tolist()]
    new_edge_index_ = torch.stack((torch.tensor(row), torch.tensor(col)), 0)
    # merged_edges.append(new_edge_index_)
    new_edge_index = torch_geometric.utils.to_dense_adj(new_edge_index_).squeeze(0)

    # subadjs.append(sp.csr_matrix(new_edge_index))
    subset_nodes_new = np.unique(new_edge_index_)
    return new_edge_index_, subset_nodes_new
import utils.aug as aug
def augmentation(subfeatures,adj,aug_type,drop_edge=0.2):
    if aug_type == 'edge':
        aug_features1 = subfeatures

        aug_adj1 = aug.aug_random_edge(adj, drop_edge=drop_edge)  # random drop edges

    elif aug_type == 'node':

        aug_features1, aug_adj1 = aug.aug_drop_node(subfeatures, adj, drop_edge=drop_edge)
        aug_features2, aug_adj2 = aug.aug_drop_node(subfeatures, adj, drop_edge=drop_edge)

    elif aug_type == 'subgraph':

        aug_features1, aug_adj1 = aug.aug_subgraph(subfeatures, adj, drop_edge=drop_edge)
        aug_features2, aug_adj2 = aug.aug_subgraph(subfeatures, adj, drop_edge=drop_edge)

    elif aug_type == 'mask':

        aug_features1 = aug.aug_random_mask(subfeatures, drop_edge=drop_edge)
        # aug_features2 = None

        aug_adj1 = adj
        # aug_adj2 = adj

    else:
        assert False
    return aug_features1, aug_adj1

def indexing_negative_samples(subfea,subset_nodes_new,subfea_index,negative_indices,negative_fea,negative_fea_count,negative_subgraph):
    for no, node in enumerate(subset_nodes_new):  # reconstruct node feature by new labels
        if node not in subfea_index:
            subfea_index += [node]
            negative_fea += [subfea[no].unsqueeze(0)]
            negative_fea_count += 1
            negative_indices += [negative_subgraph]

        else:
            print(node)
    return subfea_index, negative_indices,negative_fea,negative_fea_count,negative_subgraph

from tqdm.contrib import tzip
def get_negative_subgraphs(sample_num,subfeatures,subgraphs,merged_edges,aug_type='mask',drop_edge=0.2):

    negative_indices = []
    count_nodes = 0
    subgraphs_ = []
    merged_edges_ = []
    negative_fea_count = 0
    subfea_index = []
    negative_fea = []
    negative_subgraph = 0

    for num, (subfea, subnodes) in enumerate(tzip(subfeatures, subgraphs)): #subgraphs:subgraph consists self node, hop1 neighbors, hop2 neighbors

        subfea = torch.index_select(subfeatures,0,torch.tensor(subnodes))
        rest = [id for id in range(len(subgraphs)) if id != num]
        random.shuffle(rest)
        #Original subgraph
        if len(subnodes) > 1:
            new_edge_index, subset_nodes_new = get_new_edge_index(list(subnodes), merged_edges, count_nodes)  # relabel original subgraphs: node and edge index
            subgraphs_.append(subset_nodes_new)
            merged_edges_.append(new_edge_index)
            count_nodes += len(subset_nodes_new)
            if len(negative_indices) != 0:
                negative_subgraph += 1
            subfea_index, negative_indices,negative_fea,negative_fea_count,negative_subgraph = indexing_negative_samples(subfea,subset_nodes_new, subfea_index, negative_indices, negative_fea,
                                      negative_fea_count, negative_subgraph)
            if aug_type =='edge':
                new_edge_index, subset_nodes_new = get_new_edge_index(list(subnodes), merged_edges,count_nodes) #relabel
                new_edge_index_matrix =torch_geometric.utils.to_dense_adj(new_edge_index).squeeze(0)
                assert len(subset_nodes_new) == len(subnodes)
                count_nodes += len(subset_nodes_new)
                subfea_, adj_ = augmentation(subfea, new_edge_index_matrix, aug_type=aug_type, drop_edge=drop_edge) #change adj to get positive augmentation samples

                subgraphs_.append(subset_nodes_new)
                merged_edges_.append(adj_)

                negative_subgraph += 1
                subfea_index, negative_indices, negative_fea, negative_fea_count, negative_subgraph = indexing_negative_samples(
                    subfea,
                    subset_nodes_new, subfea_index, negative_indices, negative_fea,
                    negative_fea_count, negative_subgraph)
            if aug_type == 'mask':
                new_edge_index, subset_nodes_new = get_new_edge_index(list(subnodes), merged_edges, count_nodes)
                count_nodes += len(subset_nodes_new)
                assert len(subset_nodes_new) == len(subnodes)
                subgraphs_.append(subset_nodes_new)
                merged_edges_.append(new_edge_index)

                fea_, adj_ = augmentation(subfea, None, aug_type=aug_type, drop_edge=drop_edge)
                negative_subgraph += 1
                # negative_adj.append(adj_)
                subfea_index, negative_indices, negative_fea, negative_fea_count, negative_subgraph = indexing_negative_samples(
                    fea_,
                    subset_nodes_new, subfea_index, negative_indices, negative_fea,
                    negative_fea_count, negative_subgraph)
                assert len(negative_indices) == len(
                    negative_fea), "Make sure the number of nodes of negative graph == node feature shape(0)"

            count = 0
            for i in rest:
                if count == sample_num:
                    break
                if len(subgraphs[i]) > 1:  # if negative sample has 2hop neighbors
                    # print(len(subgraphs[i]))

                    new_edge_index, subset_nodes_new = get_new_edge_index(list(subgraphs[i]), merged_edges,
                                                            count_nodes)  # relabel
                    count_nodes += len(subset_nodes_new)
                    assert len(subset_nodes_new) == len(subgraphs[
                                                            i]), "Need to make sure the number of nodes in a subgraph does not change after relabeling"
                    negative_subgraph += 1  # count the subgraphs

                    subgraphs_.append(subset_nodes_new)
                    merged_edges_.append(new_edge_index)

                    subfea_index, negative_indices, negative_fea, negative_fea_count, negative_subgraph = indexing_negative_samples(
                        subfeatures[i],
                        subset_nodes_new, subfea_index, negative_indices, negative_fea,
                        negative_fea_count, negative_subgraph)
                    count +=1
                else:
                    continue

            assert count == sample_num

    negative_fea_tensor = torch.cat(negative_fea, 0)
    # negative_indices = torch.tensor(negative_indices).cuda()
    merged_edges_ = torch.cat(merged_edges_, dim=1)
    return negative_fea_tensor, merged_edges_,negative_indices
