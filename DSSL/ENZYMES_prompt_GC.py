import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from torch_geometric.loader import DataLoader
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
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
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
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
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
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--sample_num', type=int, default=10, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=64, help='number of neighbors')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='dssl_no_prompt_with_metanet_origin_feature_graph', help='number of neighbors')

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
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):

    cnt = 0

    result = torch.FloatTensor(graph_sizes.shape[0], batched_graph_feats.shape[1]).cuda()

    for i in range(graph_sizes.shape[0]):
        # print("i",i)
        current_graphlen = int(graph_sizes[i].item())
        graphlen = range(cnt,cnt+current_graphlen)
        # print("graphlen",graphlen)
        result[i] = torch.sum(batched_graph_feats[graphlen,:], dim=0)
        cnt = cnt + current_graphlen
    # print("resultsum",cnt)
    return result
def evaluate1(model,train_dataset,test_dataset,train_label,test_label,test_graph_len,train_graph_len):
    model.eval()
    train_dataset.graph['node_feat'] = train_dataset.graph['node_feat'].cuda()
    train_embedding = model.online_encoder(train_dataset)
    train_embedding = train_embedding.detach()
    train_embedding = split_and_batchify_graph_feats(train_embedding, train_graph_len)

    test_embedding = model.online_encoder(test_dataset)
    test_embedding = test_embedding.detach()
    test_embedding = split_and_batchify_graph_feats(test_embedding, test_graph_len)

    emb_dim, num_class = train_embedding.shape[1], train_dataset.label.unique().shape[0]
    train_accs, dev_accs, test_accs =[], [], []

    for i in range(10):

        classifier = LogisticRegression(emb_dim, num_class).to(device)
        optimizer_LR = torch.optim.AdamW(classifier.parameters(), lr=0.01, weight_decay=0.01)

        for epoch in range(100):
            classifier.train()

            logits, loss = classifier(train_embedding, train_label.squeeze())
            # print ("finetune epoch: {}, finetune loss: {}".format(epoch, loss))
            optimizer_LR.zero_grad()
            loss.backward()
            optimizer_LR.step()

        # train_logits, _ = classifier(train_embedding, train_label.squeeze())
        # dev_logits, _ = classifier(embedding[valid_idx, :], valid_label.squeeze())

        test_logits, _ = classifier(test_embedding, test_label.type(torch.long).cuda())

        test_preds = torch.argmax(test_logits, dim=1)


        test_acc = (torch.sum(test_preds == test_label.squeeze()).float() /
                    test_label.squeeze().shape[0]).detach().cpu().numpy()


        test_accs.append(test_acc * 100)

    test_accs = np.stack(test_accs)


    test_acc, test_std = test_accs.mean(), test_accs.std()

    return test_acc,test_std

### Load and preprocess data ###
import ast
if args.dataset in ['BZR','COX2']:

    with open(f"../data/{args.dataset}/{args.dataset}.edges") as f:
        edges_index = [ast.literal_eval(i) for i in  f.readlines()]
    with open(f"../data/{args.dataset}/{args.dataset}.graph_idx") as f:
        graph_labels =[ast.literal_eval(i) for i in  f.readlines()]
    with open(f"../data/{args.dataset}/{args.dataset}.node_attrs") as f:
        features= np.array([ast.literal_eval(i) for i in f.readlines()])
    with open(f"../data/{args.dataset}/{args.dataset}.node_labels") as f:
        labels= torch.tensor([ast.literal_eval(i) for i in f.readlines()]).cuda()

    # features = process.preprocess_features(features)
    # a = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edges_index).T)
    # adj = sp.csr_matrix(a)
    # origin_adj = np.array(adj.todense())
    features = torch.FloatTensor(features).cuda()
    nb_nodes = features.size(1)
    edges_index=torch.tensor(edges_index).cuda()

    graph = dgl.graph((np.array(edges_index.cpu())[0], np.array(edges_index.cpu())[1]))
    # graph = dgl.remove_nodes(graph, torch.tensor(range(19470,19580)))
    # edge_index = torch.stack(graph.edges(),dim=0)
    graph = graph.remove_self_loop().add_self_loop()
    graph = graph.to(device)
    # from dataset import NCDataset
    dataset = NCDataset(f"ori_{args.dataset}")
    num_nodes = len(features)
    dataset.graph = {'edge_index': edges_index,
                     'node_feat': features,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.graph['edge_index'], edge_attr, mask = remove_isolated_nodes(dataset.graph['edge_index'])
if  args.dataset not in ['ENZYMES','PROTEINS','chameleon','wisconsin','BZR','COX2']:
    dataset = load_dataset(args.dataset, args.sub_dataset)

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

if args.rand_split or args.dataset in ['snap-patents','ogbn-proteins', 'wiki','Cora', 'PubMed','genius']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                for _ in range(args.runs)]

#Test set
from torch_geometric.loader import DataLoader
if args.dataset in ['ENZYMES', 'PROTEINS','cornell','chameleon','wisconsin','squirrel', 'COX2','BZR']:
    path = f"../data/fewshot_{args.dataset}_graph"
else:
    path = f"../data/fewshot_{args.dataset}"
if args.dataset in ['cornell','chameleon','squirrel','wisconsin']:
    idx_test = torch.load(f"{path}/test_idx.pt").type(torch.long).cuda()
    test_lbls = torch.load(f"{path}/test_labels.pt").type(torch.long)
    origin_datasets = torch.load(f'{path}/graph_data1.pkl')
    test_datasets = [origin_datasets[t] for t in idx_test]
    test_loader = DataLoader(test_datasets, batch_size=len(test_datasets), follow_batch=['x'] * len(test_datasets),
                             shuffle=False)


else:
    test_features = torch.load(f"{path}/testset/feature.pt")
    # test_features=process.preprocess_features(test_features)
    testfeature = torch.FloatTensor(test_features).cuda()
    test_adj = torch.load(f"{path}/testset/adj.pt")
    test_adj_ = test_adj.numpy()
    # adj_csr = sp.csr_matrix(test_adj_)
    # test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
    # sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()
    test_lbls = torch.load(f"{path}/testset/labels.pt").cuda()
    nb_classes = len(torch.unique(test_lbls))
    test_graph_len = torch.load(f"{path}/testset/graph_len.pt")

    test_edge_index = test_adj.nonzero().t().contiguous().cuda()
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

# if not os.path.exists(f"../data/fewshot_{args.dataset}/testneighbors1.csv"):
testneighbors = [[] for m in range(len(test_adj_))]
testneighbors_2hop = [[] for m in range(len(test_adj_))]
for step, x in enumerate(range(test_adj_.shape[0])):
    tempneighbors, tempneighbors_2hop =find_2hop_neighbors(test_adj_, x)
    testneighbors[step] = tempneighbors
    testneighbors_2hop[step] = tempneighbors_2hop
#     with open(f"../data/fewshot_{args.dataset}/testneighbors1.csv", "w") as f:
#         wr = csv.writer(f)
#         wr.writerows(testneighbors)
#     with open(f"../data/fewshot_{args.dataset}/testneighbors_2hop1.csv", "w") as f:
#         wr = csv.writer(f)
#         wr.writerows(testneighbors_2hop)
# # else:
#     testneighbors = []
#     file = open(f"../data/fewshot_{args.dataset}/testneighbors1.csv")
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         testneighbors.append([int(i) for i in row])
#
#     testneighbors_2hop = []
#     file = open(f"../data/fewshot_{args.dataset}/testneighbors_2hop1.csv")
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         testneighbors_2hop.append([int(i) for i in row])
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
patience=50
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum", shotnum)
    for i in range(100):
        cnt_wait = 0
        if args.dataset in ['ENZYMES', 'PROTEINS','COX2','BZR']:
            adj_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/adj.pt")
            feature_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/feature.pt")
            train_lbls = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(
                torch.long).squeeze().cuda()
            train_graph_len = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/graph_len.pt")

            pretrain_edge_index = adj_train.nonzero().t().contiguous().cuda()
            pretrain_graph = dgl.graph((np.array(pretrain_edge_index.cpu())[0], np.array(pretrain_edge_index.cpu())[1]))
            pretrain_graph = pretrain_graph.remove_self_loop().add_self_loop()
            pretrain_graph = pretrain_graph.to(device)

            train_dataset = NCDataset(f"train_{dataset}")
            num_nodes = len(feature_train)
            train_dataset.graph = {'edge_index': pretrain_edge_index,
                                  'node_feat': feature_train,
                                  'edge_feat': None,
                                  'num_nodes': num_nodes}
            train_dataset.label = train_lbls
            model.eval()
            train_embedding = model.online_encoder(train_dataset)
            train_embedding = train_embedding.detach()
            test_embedding = model.online_encoder(test_dataset)
            test_embedding = test_embedding.detach()
            adj_train_ = adj_train.numpy()


            neighbors = [[] for m in range(len(adj_train_))]
            neighbors_2hop = [[] for m in range(len(adj_train_))]
            for step, x in enumerate(range(len(adj_train_))):
                tempneighbors, tempneighbors_2hop = find_2hop_neighbors(adj_train_, x)
                neighbors[step] = tempneighbors
                neighbors_2hop[step] = tempneighbors_2hop
        embedding = model.online_encoder(dataset)
        embedding = embedding.detach()
        # print("true", i, train_lbls)
        nb_classes = len(torch.unique(test_lbls))

        log = downprompt(neighbors, neighbors_2hop, nb_classes, embedding, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction, multi_prompt=args.multi_prompt,bottleneck_size=args.bottleneck_size,
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
            logits = log.forward2(train=1, train_embeds=train_embedding,graph_len=train_graph_len).float().cuda()
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
            # logits = log(test_embs,neighbors=testneighbors,neighbors_2hop=testneighbors_2hop)#train_adj=test_adj)
            logits = log.forward2(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,
                                  test_embeds=test_embedding,graph_len=test_graph_len)  # test_embeds=test_embeds, test_embeds=test_embeds
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