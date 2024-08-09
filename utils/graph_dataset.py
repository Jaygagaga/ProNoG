import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np


import random

import dgl

from utils import process



import argparse
import csv
import torch_geometric
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset

import tqdm

parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="cornell", help='data')
parser.add_argument('--dataset_root', type=str, default="/home/xingtong/WebKB_node_origin/data", help='path to save data')
parser.add_argument('--sample_num', type=int, default=20, help='number of negative samples')

parser.add_argument('--aug_type', type=str, default="mask", help='aug type: mask or edge')
parser.add_argument('--drop_edge', type=float, default=0.2, help='drop percent')

# pretraining hyperperemeter
parser.add_argument('--k1', type=int, default=10, help='number of hop-2 neighbors')
parser.add_argument('--k2', type=int, default=20, help='number of hop-3 neighbors')





args = parser.parse_args()

if os.path.exists(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph") == False:
    os.makedirs(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph")


import torch



sparse = True
useMLP = False

nonlinearity = 'prelu'  # special name to separate parameters

from utils.heterophilic import  WikipediaNetwork, Actor
from torch_geometric.datasets import HeterophilousGraphDataset
if args.dataset in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
    datasets = HeterophilousGraphDataset(root=args.dataset_root, name=args.dataset)
# if args.dataset =='cornell':
if args.dataset in ['cornell', 'texas', 'wisconsin']:
    datasets = WebKB(root=args.dataset_root, name=args.dataset)
    # features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # idx_test = range(features.shape[0]-100,features.shape[0])
    # origin_adj = adj.todense().A
if args.dataset in ['chameleon', 'squirrel']:
    datasets = WikipediaNetwork(root=args.dataset_root, name=args.dataset)
# if args.dataset in ['film']:
#     datasets = Actor(root='data')
features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
# features= process.preprocess_features(features)
# features = torch.FloatTensor(features[np.newaxis])
# features = features.cuda()
labels = torch.FloatTensor(labels[np.newaxis]).cuda()
# idx_test = torch.load(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/test_idx.pt").type(torch.long).cuda()
# test_lbls = torch.load(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/test_labels.pt").type(torch.long).cuda()
if args.dataset == 'cornell':
    origin_adj = adj.todense().A.T
else:
    origin_adj = adj.todense().A

"""
------------------------------------------------------------
2hop neighbors and subgraphs and graph data
------------------------------------------------------------
"""
neighbors = []
neighbors_2hops = []
neighbors_3hops = []
neighborslist = [[] for x in range(adj.shape[0])]
neighbors_2hoplist = [[] for x in range(adj.shape[0])]
neighbors_3hoplist = [[] for x in range(adj.shape[0])]
neighborsindex = [[] for x in range(adj.shape[0])]
neighbors_2hopindex = [[] for x in range(adj.shape[0])]
neighbors_3hopindex = [[] for x in range(adj.shape[0])]

neighboradj = adj.todense().A
for x in tqdm.trange(adj.shape[0]):
    neighborslist[x], neighbors_2hoplist[x],neighbors_3hoplist[x]= process.find_3hop_neighbors(origin_adj,x,args.k1,args.k2)
    temp1 = [x] *len(neighborslist[x])
    temp2 = [x] *len(neighbors_2hoplist[x])
    temp3 = [x] * len(neighbors_3hoplist[x])
    # print(temp1)
    neighbors.append(neighborslist[x])
    neighbors_2hops.append(neighbors_2hoplist[x])
    neighbors_3hops.append(neighbors_3hoplist[x])
    neighborsindex[x] = temp1
    neighbors_2hopindex[x] = temp2
    neighbors_3hopindex[x] = temp3
neighborslist = sum(neighborslist,[])
neighbors_2hoplist = sum(neighbors_2hoplist,[])
neighbors_3hoplist = sum(neighbors_3hoplist,[])
neighborsindex = sum(neighborsindex,[])
neighbors_2hopindex = sum(neighbors_2hopindex,[])
neighbors_3hopindex = sum(neighbors_3hopindex,[])

neighbortensor = torch.zeros(len(neighborslist),adj.shape[0])
neighbors_2hoptensor = torch.zeros(len(neighbors_2hoplist),adj.shape[0])
neighbors_3hoptensor = torch.zeros(len(neighbors_3hoplist),adj.shape[0])

for x in tqdm.trange(len(neighborslist)):
    neighbortensor[x][neighborslist[x]] = 1

for x in tqdm.trange(len(neighbors_2hoplist)):
    neighbors_2hoptensor[x][neighbors_2hoplist[x]] = 1

for x in tqdm.trange(len(neighbors_3hoplist)):
    neighbors_3hoptensor[x][neighbors_3hoplist[x]] = 1
with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/neighbors.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(neighbors)
with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/neighbors_2hops.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(neighbors_2hops)
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
edge_index = from_scipy_sparse_matrix(adj)[0]
graph_labels = []
graph_list = []
graph_list1 = []
count = 0
for x in tqdm.trange(origin_adj.shape[0]):

    graph_label = torch.argmax(labels[0,x])
    subgraph_nodes = list(np.unique([x]+neighbors[x]+neighbors_2hops[x]+neighbors_3hops[x]))
    if len(subgraph_nodes) !=1:

        new_edge_index= torch_geometric.utils.subgraph(edge_index=edge_index, subset=subgraph_nodes,
                                                           relabel_nodes=True)[0]
        old_edge_index = torch_geometric.utils.subgraph(edge_index=edge_index, subset=subgraph_nodes,
                                                        relabel_nodes=False)[0]
        dicts = {i:j for i,j in zip(list(np.unique(old_edge_index[0].tolist()+old_edge_index[1].tolist())),list(np.unique(new_edge_index[0].tolist()+new_edge_index[1].tolist())))}

        # if x in old_edge_index[0].tolist():
        #     idx = old_edge_index[0].tolist().index(x)
        #     center_node =new_edge_index[0][idx].item()
        # else:
        #     idx = old_edge_index[1].tolist().index(x)
        #     center_node = new_edge_index[1][idx].item()
        node_feature = torch.index_select(features,0,torch.tensor(subgraph_nodes))
        data = dgl.graph((np.array(new_edge_index)[0], np.array(new_edge_index)[1]))
        data.ndata['x'] = node_feature
        graph_list.append(data)
        graph_labels.append(graph_label)
        data1 = Data(x=node_feature,edge_index=new_edge_index,y=graph_label,graph_idx=count,graph_nodes=[subgraph_nodes])
        graph_list1.append(data1)
        count +=1
torch.save(graph_list1,f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data1.pkl')


'''
------------------------------------------------------------
negative sampling
------------------------------------------------------------
'''
from utils.process import augmentation
from tqdm.contrib import tzip
# graph_datasets= []
graph_datasets1=[]
# graph_datasets_labels= []
graph_count = 0
for num, (graph, graph1) in enumerate(tzip(graph_list,graph_list1)):
    x = graph1.x
    edge_index = graph1.edge_index
    y = graph1.y
    graph_idx = num*(args.sample_num+2)
    # graph_count +=1
    data = graph1
    data.graph_idx = graph_idx
    rest = [id for id in range(len(graph_list1)) if id != num]
    negative_samples = []
    negative_samples1 = []
    random.shuffle(rest)
    # if args.aug_type =='m':
    new_edge_index_matrix = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
    subfea_, adj_ = augmentation(x, new_edge_index_matrix, aug_type=args.aug_type,drop_edge=0.2)  # change adj to get positive augmentation samples
    # dgl data
    # edge_index_ = from_scipy_sparse_matrix(adj_)[0] torch_geometric.utils.to_edge_index(adj_)
    positive_sample = dgl.graph((np.array(adj_)[0], np.array(adj_)[1]))
    if args.aug_type == 'edge':
        # new_nodes = positive_sample.nodes()
        positive_sample.ndata['x'] = graph.ndata['x'][positive_sample.nodes()]
        positive_sample1 = Data(x=graph.ndata['x'][positive_sample.nodes()], edge_index=adj_, y=y,graph_idx=graph_idx+1)
    if args.aug_type == 'mask':
        # positive_sample.ndata['x'] = subfea_
        positive_sample1 = Data(x=subfea_, edge_index=adj_, y=y,graph_idx=graph_idx+1)
    # graph_datasets.append(positive_sample)
    # graph_datasets.append(graph)
    # graph_datasets_labels +=[y]*2
    # torch_geometric Dataset
    graph_datasets1.append(data)
    graph_datasets1.append(positive_sample1)
    count = 0
    for i in rest:
        if count == args.sample_num:
            break
        # negative_samples.append(graph_list[i])
        data = graph_list1[i]
        data.graph_idx = graph_idx+1+1+count
        negative_samples1.append(data)
        count +=1
        # graph_datasets_labels.append(graph_list1[i].y)
    assert len(negative_samples1) ==args.sample_num
    # graph_datasets += negative_samples
    graph_datasets1 += negative_samples1

torch.save(graph_datasets1,f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data_augmented_{args.sample_num}_{args.aug_type}.pkl')
print('End')
# gdl_graphs = dgl.batch(graph_datasets)
# dataloader = DataLoader(graph_datasets1,batch_size=256)












