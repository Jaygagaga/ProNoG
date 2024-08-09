import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from torch_geometric.loader import DataLoader
import torch_geometric.utils
from torch_scatter import scatter
from torch_geometric.utils import to_undirected, remove_isolated_nodes
torch.autograd.set_detect_anomaly(True)
from logger import Logger
from dataset import load_dataset,Large_Dataset
from data_utils import evaluate, eval_acc,to_sparse_tensor, eval_rocauc, sample_neighborhood,load_fixed_splits, sample_neg_neighborhood
from encoders import LINK, GCN, MLP, SGC, GAT, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net, LINK_Concat, LINKX, GPRGNN, GCNII
import faulthandler
from downprompt_metanet3 import downprompt
faulthandler.enable()
# import process
from models import DSSL,LogisticRegression
from os import path
import os
DATAPATH = path.dirname(path.dirname(path.abspath(__file__))) + '/data/' \
### Parse args ###
from heterophilic import  WikipediaNetwork, Actor
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.datasets import WebKB
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dataset', type=str, default='ENZYMES')
parser.add_argument('--save_name',        type=str,           default='../modelset/dssl/ENZYMES_DSSL.pkl',                help='save ckpt name')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')

parser.add_argument('--sub_dataset', type=str, default='DE')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=int, default=0)# 0.01
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--directed', action='store_true', help='set to not symmetrize adjacency')
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=39, help='Random seed.')
parser.add_argument('--display_step', type=int, default=25, help='how often to print')
parser.add_argument('--train_prop', type=float, default=.48, help='training label proportion')
parser.add_argument('--valid_prop', type=float, default=.32, help='validation label proportion')
parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
parser.add_argument('--rand_split', type=bool, default=True, help='use random splits')
parser.add_argument('--embedding_dim', type=int, default=10, help="embedding dim")
parser.add_argument('--neighbor_max', type=int, default=5, help="neighbor num max")
parser.add_argument('--cluster_num', type=int, default=6, help="cluster num")
parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--entropy', type=float, default=0.0)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--encoder', type=str, default='MLP')
parser.add_argument('--mlp_bool', type=int, default=1, help="embedding with mlp predictor")
parser.add_argument('--tao', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--mlp_inference_bool', type=int, default=1, help="embedding with mlp predictor")
parser.add_argument('--neg_alpha', type=int, default=0, help="negative alpha ")
parser.add_argument('--load_json', type=int, default=0, help="load json")
parser.add_argument('--use_origin_feature', type=int, default=1, help='')
parser.add_argument('--prompt', type=int, default=0, help='0:no prompt,1:use prompt')
parser.add_argument('--multi_prompt', type=int, default=0, help='1:multi prompt or 0:single prompt')
parser.add_argument('--use_metanet', type=int, default=1, help='use metanet layer or not')
parser.add_argument('--meta_in', type=int, default=64, help='hidden size of metanet input')
parser.add_argument('--bottleneck_size', type=int, default=4, help='bottleneck size')
parser.add_argument('--out_size', type=int, default=64, help='hidden size of metanet output')
parser.add_argument('--gp', type=int, default=1, help='0: no subgraph, 1: subgraph')
parser.add_argument('--weight', type=int, default=1, help='0: no weights, 1: weights')
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--pretrain_hop', type=int, default=0,  help='pretrain_hop')
parser.add_argument('--concat_dense', type=int, default=0, help='1: apply concat then linear')
parser.add_argument('--down_weight', type=int, default=0, help='0: no downstream weights, 1: weights')
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--sample_num', type=int, default=10, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=64, help='number of neighbors')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='dssl_no_prompt_with_metanet_origin_feature', help='number of neighbors')

args = parser.parse_args()
print(args)
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


if args.lr == 0:
    args.lr = 0.001
elif args.lr == 1:
    args.lr = 0.01


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

def extract_args_from_json(json_file_path,args_dict):

    import json
    summary_filename = 'json/'+json_file_path+'.json'
    import os
    if os.path.isfile(summary_filename):
        with open(summary_filename) as f:
            summary_dict = json.load(fp=f)
        for key in summary_dict.keys():
            args_dict[key] = summary_dict[key]
        return args_dict
    return args_dict

print(args.dataset)
args_dict = vars(args)
if args.load_json:
    args_dict = extract_args_from_json(args.dataset, args_dict)
    args = Bunch(args_dict)

### Seeds ###
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import sys

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    if dataset_str == 'CiteSeer':
        dataset_str = 'citeseer'
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
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
### device ###
import networkx as nx
import dgl
# from hetero_dataset import WikipediaNetwork
# from torch_geometric.datasets import WebKB, Actor
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from dataset import NCDataset
device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

### Load and preprocess data ###
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
if  args.dataset not in ['ENZYMES','PROTEINS', 'chameleon','wisconsin','BZR','COX2'] :
    dataset = load_dataset(args.dataset, args.sub_dataset)
    origin_adj = torch_geometric.utils.to_dense_adj(dataset.graph['edge_index']).squeeze(0).numpy()
if args.dataset in ['chameleon','wisconsin']:
    if args.dataset in ['chameleon']:
        datasets = WikipediaNetwork(root='../data', name=args.dataset)
    if args.dataset in ['wisconsin']:
        datasets = WebKB(root='../data', name=args.dataset)
    features, adj, labels, sizes, masks = process_webkb(datasets.data, datasets.data.x.shape[0])
    # features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # idx_test = range(features.shape[0]-100,features.shape[0])
    origin_adj = adj.todense().A
    if args.dataset == 'cornell':
        origin_adj = adj.todense().A.T
    edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
    dataset = NCDataset(args.dataset)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.size(0)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(labels, dtype=torch.long)
    dataset.label = label
if  args.dataset in ['ENZYMES','PROTEINS'] :
    data = torch.load(f"../data/{args.dataset}/data.pkl")
    features = data.features
    features = torch.tensor(features).cuda()
    ft_size = features.shape[-1]
    adj = data.adj
    origin_adj = np.array(adj.todense())
    labels = data.labels
    labels_ = torch.argmax(torch.tensor(labels), dim=-1)
    edge_index = data.edge_index
    graph = dgl.graph((np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1]))
    # graph = dgl.remove_nodes(graph, torch.tensor(range(19470,19580)))
    # edge_index = torch.stack(graph.edges(),dim=0)
    graph = graph.remove_self_loop().add_self_loop()
    graph = graph.to(device)
    # from dataset import NCDataset
    dataset = NCDataset(f"ori_{args.dataset}")
    num_nodes = len(features)
    dataset.graph = {'edge_index': edge_index,
                          'node_feat': features,
                          'edge_feat': None,
                          'num_nodes': num_nodes}
    dataset.label = labels_
    dataset.graph['edge_index'], edge_attr, mask = remove_isolated_nodes(dataset.graph['edge_index'])
    nb_classes = dataset.label.unique().shape[0]
    # features_ = torch.FloatTensor(features[np.newaxis]).cuda()
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    # sp_adj_ = process.sparse_mx_to_torch_sparse_tensor(adj)
    # sp_adj_ = sp_adj_.to_dense().cuda()
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.dataset in ['snap-patents', 'CiteSeer','genius']:
    dataset.graph['edge_index'],edge_attr, mask = remove_isolated_nodes(dataset.graph['edge_index'])
    dataset.graph['node_feat']= dataset.graph['node_feat'][mask]
    dataset.label= dataset.label[mask]

if args.rand_split or args.dataset in ['snap-patents','ogbn-proteins', 'wiki','cora', 'PubMed','genius']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                for _ in range(args.runs)]

#Test set
if args.dataset in ['cornell', 'texas', 'wisconsin','chameleon','squirrel']:
    idx_test = torch.load(f"../data/fewshot_{args.dataset}/test_idx.pt").type(
        torch.long).cuda()
    test_lbls = torch.load(f"../data/fewshot_{args.dataset}/test_labels.pt").type(
        torch.long).cuda()
elif  args.dataset in ['ENZYMES','PROTEINS']:
    if args.dataset == 'ENZYMES':
        test_adj = torch.load("../data/fewshot_ENZYMES/5-shot_ENZYMES/testadj.pt").squeeze().cuda()
        testfeature = torch.load("../data/fewshot_ENZYMES/5-shot_ENZYMES/testemb.pt").squeeze(0).cuda()
        test_lbls = torch.load("../data/fewshot_ENZYMES/5-shot_ENZYMES/testlabels.pt").type(torch.long).squeeze().cuda()
    if args.dataset == 'PROTEINS':
        test_adj = torch.load(
            "../data/fewshot_PROTEINS/1-shot_PROTEINS/testadj.pt").squeeze().cuda()
        testfeature = torch.load(
            "../data/fewshot_PROTEINS/1-shot_PROTEINS/testemb.pt").squeeze(0).cuda()
        test_lbls = torch.load(
            "../data/fewshot_PROTEINS/1-shot_PROTEINS/testlabels.pt").type(
            torch.long).squeeze().cuda()
    test_adj_ = test_adj.cpu().numpy()
    test_edge_index = test_adj.nonzero().t().contiguous()
    testgraph = dgl.graph((np.array(test_edge_index.cpu())[0], np.array(test_edge_index.cpu())[1]))
    testgraph = testgraph.remove_self_loop().add_self_loop()
    testgraph = testgraph.to(device)
    # from dataset import NCDataset
    test_dataset = NCDataset(f"test_{dataset}")
    num_nodes= len(testfeature)
    test_dataset.graph = {'edge_index': test_edge_index,
                     'node_feat': testfeature,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    test_dataset.label = test_lbls
    test_dataset.graph['edge_index'], edge_attr, mask = remove_isolated_nodes(test_dataset.graph['edge_index'])

else:
    if args.dataset =='CiteSeer':
        adj_node, features, labels_node, idx_train, idx_val, idx_test = load_data("citeseer")
    else:
        adj_node, features, labels_node, idx_train, idx_val, idx_test = load_data(args.dataset)
    test_lbls=torch.tensor(labels_node[idx_test]).cuda()
    test_lbls = torch.argmax(test_lbls, dim=-1)

if args.dataset =='CiteSeer':
    path = f"../data/fewshot_citeseer"
elif args.dataset =='cora':
    path = f"../data/fewshot_cora"
else:
    path = f"../data/fewshot_{args.dataset}"

dataset.graph['num_nodes']=dataset.graph['node_feat'].shape[0]

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])


dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

dataset.label = dataset.label.to(device)


sampled_neighborhoods = sample_neighborhood(dataset, device, args)

if args.neg_alpha:
    sampled_neg_neighborhoods = sample_neg_neighborhood(dataset, device, args)
    print('sample_neg_neighborhoods')


### Choose encoder ###

if args.encoder == 'GCN':
    encoder = GCN(in_channels=d,
                  hidden_channels=args.hidden_channels,
                  out_channels=args.hidden_channels,
                  num_layers=args.num_layers, use_bn=not args.no_bn,
                  dropout=args.dropout).to(device)
else:
    encoder = MLP(in_channels=d,
                  hidden_channels=args.hidden_channels,
                  out_channels=args.hidden_channels,
                  num_layers=args.num_layers,
                  dropout=args.dropout).to(device)

model = DSSL(encoder=encoder,
             hidden_channels=args.hidden_channels,
             dataset=dataset,
             device=device,
             cluster_num=args.cluster_num,
             alpha=args.alpha,
             gamma=args.gamma,
            tao=args.tao,
            beta=args.beta,
             moving_average_decay=args.tau).to(device)

if not args.mlp_bool: # 0 embedding without mlp predictor
    model.Embedding_mlp = False
if not args.mlp_inference_bool: # 0 embedding without mlp predictor
    model.inference_mlp = False
### logger ###
logger = Logger(args.runs, args)
import pandas as pd
import os
model.train()
print('MODEL:', model)

# print (split_idx_lst)
import datetime

time_now = datetime.datetime.now()
# print('start training')
print(time_now)
xent = nn.CrossEntropyLoss()
meanAcc = 0
import csv

import csv
if args.dataset not in ['PROTEINS', 'ENZYMES']:
    neighbors = []
    file = open(f"{path}/neighbors.csv")
    csvreader = csv.reader(file)
    for row in csvreader:
        neighbors.append([int(i) for i in row])
    neighbors_2hops = []
    file = open(f"{path}/neighbors_2hops.csv")
    csvreader = csv.reader(file)
    for row in csvreader:
        neighbors_2hops.append([int(i) for i in row])

    testneighbors = [neighbors[i] for i in idx_test]
    testneighbors_2hop = [neighbors_2hops[i] for i in idx_test]
else:
    if not os.path.exists(f"../data/fewshot_{args.dataset}/testneighbors1shot.csv"):
        testneighbors = [[] for m in range(len(test_adj_))]
        testneighbors_2hop = [[] for m in range(len(test_adj_))]
        for step, x in enumerate(range(test_adj_.shape[0])):
            tempneighbors, tempneighbors_2hop = find_2hop_neighbors(test_adj_, x)
            testneighbors[step] = tempneighbors
            testneighbors_2hop[step] = tempneighbors_2hop
        with open(f"../data/fewshot_{args.dataset}/testneighbors1shot.csv", "w") as f:
            wr = csv.writer(f)
            wr.writerows(testneighbors)
        with open(f"../data/fewshot_{args.dataset}/testneighbors_2hop1shot.csv",
                  "w") as f:
            wr = csv.writer(f)
            wr.writerows(testneighbors_2hop)
    else:
        testneighbors = []
        file = open(f"../data/fewshot_{args.dataset}/testneighbors1shot.csv")
        csvreader = csv.reader(file)
        for row in csvreader:
            testneighbors.append([int(i) for i in row])

        testneighbors_2hop = []
        file = open(f"../data/fewshot_{args.dataset}/testneighbors_2hop1shot.csv")
        csvreader = csv.reader(file)
        for row in csvreader:
            testneighbors_2hop.append([int(i) for i in row])

if not os.path.exists(args.save_name):
    ## Training loop ###
    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float('-inf')
        loss_lst = []
        best_loss = float('inf')

        for epoch in range(args.epochs):
            # pre-training
            model.train()
            batch_size = args.batch_size
            perm = torch.randperm(n)
            epoch_loss = 0
            for batch in range(0, n, batch_size):
                optimizer.zero_grad()
                online_embedding = model.online_encoder(dataset)
                target_embedding = model.target_encoder(dataset)
                batch_idx = perm[batch:batch + batch_size]  # perm[2708,]
                batch_idx = batch_idx.to(device)
                batch_neighbor_index = sampled_neighborhoods[batch_idx].type(torch.long)
                batch_embedding = online_embedding[batch_idx].to(device)
                batch_embedding = F.normalize(batch_embedding, dim=-1, p=2)
                batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index.cpu()]
                batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                main_loss, context_loss, entropy_loss, k_node = model(batch_embedding, batch_neighbor_embedding)
                tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=args.cluster_num).type(torch.FloatTensor).to(
                    device)
                batch_sum = (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                if args.neg_alpha:
                    batch_neg_neighbor_index = sampled_neg_neighborhoods[batch_idx]
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neg_neighbor_index]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                    batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                    main_neg_loss, tmp, tmp, tmp = model(batch_embedding, batch_neighbor_embedding)
                    loss = main_loss + args.gamma * (context_loss + entropy_loss) + main_neg_loss

                else:
                    loss = main_loss+ args.gamma*(context_loss+entropy_loss)
                print("run : {}, batch : {}, main_loss: {}, context_loss: {}, entropy_loss: {}".format(run,batch,main_loss, context_loss, entropy_loss))
                loss.backward()
                optimizer.step()
                model.update_moving_average()
                epoch_loss = epoch_loss + loss
            if epoch %1== 0:
                model.eval()
                for batch in range(0, n, batch_size):
                    online_embedding = model.online_encoder(dataset).detach().cpu()
                    target_embedding = model.target_encoder(dataset).detach().cpu()
                    batch_idx = perm[batch:batch + batch_size]
                    batch_idx = batch_idx.to(device)
                    batch_neighbor_index = sampled_neighborhoods[batch_idx].type(torch.long)
                    batch_target_embedding = target_embedding[batch_idx.cpu()].to(device)
                    batch_embedding = online_embedding[batch_idx.cpu()].to(device)
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index.cpu()]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                    main_loss, context_loss, entropy_loss, k_node = model(batch_embedding, batch_neighbor_embedding)
                    tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=args.cluster_num).type(torch.FloatTensor).to(
                        device)
                    if batch == 0:
                        cluster = torch.matmul(batch_embedding.t(),tmp )
                        batch_sum=(torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                    else:
                        cluster+=torch.matmul(batch_embedding.t(),tmp)
                        batch_sum += (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                cluster = F.normalize(cluster, dim=-1, p=2)
                model.update_cluster(cluster,batch_sum)
            print("epoch: {}, loss: {}".format(epoch, epoch_loss))

    torch.save(model.state_dict(), args.save_name)
model.load_state_dict(torch.load(args.save_name))
patience=20
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum", shotnum)
    for i in range(100):
        cnt_wait = 0
        if args.dataset in ['ENZYMES', 'PROTEINS']:
            idx_train = torch.load(
                f"../data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodeidx.pt").type(
                torch.long).cuda()
            train_lbls = torch.load(
                f"../data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodelabels.pt").type(
                torch.long).squeeze().cuda()
            # print(len(idx_train))
            pretrain_adj = torch.load(
                f"../data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodeadj.pt").squeeze().cuda()
            prefeature = torch.load(
                f"../data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodeemb.pt").cuda()

            pretrain_edge_index = pretrain_adj.nonzero().t().contiguous().cuda()
            pretrain_graph = dgl.graph((np.array(pretrain_edge_index.cpu())[0], np.array(pretrain_edge_index.cpu())[1]))
            pretrain_graph = pretrain_graph.remove_self_loop().add_self_loop()
            pretrain_graph = pretrain_graph.to(device)

            train_dataset = NCDataset(f"train_{dataset}")
            num_nodes = len(prefeature)
            train_dataset.graph = {'edge_index': pretrain_edge_index,
                                  'node_feat': prefeature,
                                  'edge_feat': None,
                                  'num_nodes': num_nodes}
            train_dataset.label = train_lbls
            model.eval()
            train_embedding = model.online_encoder(train_dataset)
            train_embedding = train_embedding.detach()
            test_embedding = model.online_encoder(test_dataset)
            test_embedding = test_embedding.detach()
            embedding = model.online_encoder(dataset)
            embedding = embedding.detach()

        else:
            # if args.dataset in ['ENZYMES', 'PROTEINS']:
            #     idx_train = torch.load(
            #         f"{path}/{shotnum}-shot_{args.dataset}/{i}/nodeidx.pt").type(
            #         torch.long).cuda()
            #     # print('idx_train', idx_train)
            #     train_lbls = torch.load(
            #         f"{path}/{shotnum}-shot_{args.dataset}/{i}/nodelabels.pt").type(
            #         torch.long).squeeze().cuda()
            # else:

            idx_train = torch.load(
                f"{path}/{shotnum}-shot_{args.dataset}/{i}/idx.pt").type(
                torch.long).cuda()
            # print('idx_train', idx_train)
            train_lbls = torch.load(
                f"{path}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(
                torch.long).squeeze().cuda()
            embedding = model.online_encoder(dataset)
            embedding = embedding.detach()
            train_embedding = embedding.unsqueeze(0)[0,idx_train]
            test_embedding = embedding.unsqueeze(0)[0,idx_test]
            # print("true", i, train_lbls)
            if args.dataset in ['cornell','cora','CiteSeer','texas']:
                nb_classes = dataset.label.unique().shape[0]
            elif args.dataset in ['squirrel']:
                nb_classes = len(torch.unique(dataset.label))
            else:
                nb_classes = dataset.label.shape[1]
        neighbors_2hop = [[] for m in range(len(idx_train))]
        neighbors = [[] for m in range(len(idx_train))]
        # print(neighbors)
        # train_adj = adj.todense().A
        # train_adj = train_adj[:,train_range][train_range,:].A
        for step, x in enumerate(idx_train):
            tempneighbors, tempneighbors_2hop = find_2hop_neighbors(origin_adj, idx_train[step].item())
            neighbors[step] = tempneighbors
            neighbors_2hop[step] = tempneighbors_2hop

        log = downprompt(neighbors, neighbors_2hop, nb_classes, embedding, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction, multi_prompt=args.multi_prompt,
                         concat_dense=None, bottleneck_size=args.bottleneck_size,
                         meta_in=args.meta_in, out_size=args.out_size,

                         hidden_size=args.hid_units, prompt=args.prompt, use_metanet=args.use_metanet)

        # if args.dataset in ['ENZYMES', 'PROTEINS']:
        #
        #     test_acc, test_std=evaluate1(model,train_dataset,test_dataset,train_lbls,test_lbls)
        # else:
        #     test_acc, test_std=evaluate(model,dataset,idx_train,idx_test,train_lbls,test_label = test_lbls) #model,dataset,train_idx,test_idx,train_label,test_label=None
        opt = torch.optim.Adam(log.parameters(), lr=0.001)
        log.cuda()
        best = 1e9
        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(200):

            log.train()
            opt.zero_grad()

            # logits = log(pretrain_embs,1).float().cuda()
            if args.dataset in ['ENZYMES', 'PROTEINS']:
                logits = log.forward2(train=1,train_embeds=train_embedding,)
            else:
                logits = log.forward(train=1, idx=idx_train).float().cuda()
            loss = xent(logits, train_lbls)

            if loss < best:
                best = loss
                # best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), args.save_name)
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward(retain_graph=True)
            opt.step()

        log.eval()
        with torch.no_grad():
            if args.dataset in ['ENZYMES', 'PROTEINS']:
                logits = log.forward2(train=0,test_embeds=test_embedding,neighbors=testneighbors,neighbors_2hop=testneighbors_2hop)#train_adj=test_adj)
            else:
                logits = log.forward(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,
                                  idx=torch.tensor(idx_test).cuda())  # test_embeds=test_embeds, test_embeds=test_embeds
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls.cuda()).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            print('acc:[{:.4f}]'.format(acc))
            tot += acc

    print('-' * 100)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)
    df = pd.DataFrame(list(
        zip( [args.save_name], [args.hop_level],
            [args.multi_prompt], [args.prompt], [args.use_metanet], [args.bottleneck_size],
            [accs.mean().item()], [accs.std().item()], [shotnum])),
        columns=['pretrained_model', 'hop_level',   'multi_prompt',
                 'prompt', 'metanet', 'bottleneck', 'accuracy', 'mean', 'shotnum'])
    if os.path.exists(f"../data/{args.dataset}/{args.filename}.csv"):
        df.to_csv(f"../data/{args.dataset}/{args.filename}.csv", mode='a', header=False)
    else:
        df.to_csv(f"../data/{args.dataset}/{args.filename}.csv")