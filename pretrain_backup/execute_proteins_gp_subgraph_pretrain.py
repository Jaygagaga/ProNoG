import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import scipy.sparse as sp

import random
# from models import LogReg
from preprompt import PrePrompt
import preprompt
from utils import process
import pandas as pd
from downprompt_metanet3 import downprompt
import os

import argparse

import csv


parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="PROTEINS", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
# parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--gp', type=int, default=1, help='subgraph or not')
parser.add_argument('--save_name', type=str, default='/home/xingtong/WebKB_node_origin/modelset/gp/PROTEINS_256_layer1_subgraph4.pkl', help='save ckpt name')
parser.add_argument('--weight', type=int, default=0, help='0: no weights, 1: weights')


parser.add_argument('--local_rank', type=str, help='local rank for dist')
parser.add_argument('--use_origin_feature', type=int, default=1, help='')


parser.add_argument('--num_layers', type=int, default=1, help='num of layers')
parser.add_argument('--model_name', type=str, default='GCN', help='save ckpt name')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
# downstream hyperperemeter

parser.add_argument('--prompt', type=int, default=0, help='0:no prompt,1:use prompt')
parser.add_argument('--multi_prompt', type=int, default=0, help='1:multi prompt or 0:single prompt')
parser.add_argument('--use_metanet', type=int, default=1, help='use metanet layer or not')
parser.add_argument('--meta_in', type=int, default=256, help='number of neighbors')
parser.add_argument('--bottleneck_size', type=int, default=4, help='number of neighbors')
parser.add_argument('--out_size', type=int, default=256, help='number of neighbors')
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--pretrain_hop', type=int, default=0, help='pretrain_hop')
parser.add_argument('--concat_dense', type=int, default=0, help='1: apply concat then linear')
parser.add_argument('--down_weight', type=int, default=0, help='0: no downstream weights, 1: weights')
parser.add_argument('--shotnum', type=list, default=[1,2,3,4,5,6,7,8,9,10], help='shot num')
parser.add_argument('--sample_num', type=int, default=10, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=256, help='number of neighbors')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='no_prompt_with_metanet_origin_feature_gp_GCN_graph',
                    help='number of neighbors')
args = parser.parse_args()


print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
# aug_type = args.aug_type
# drop_percent = args.drop_percent
seed = args.seed
random.seed(seed)
np.random.seed(seed)
import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))
device_ids = [0, 1, 2, 3]
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
# from downprompt_metanet3 import downprompt
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# training params

# idx_train = torch.load("data/fewshot/0/idx.pt").type(torch.long).cuda()
batch_size = 8
nb_epochs = 1000
test_times = 50
#test_times = 10
patience = 10
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0


useMLP =False
class_num = 3
LP = True


nonlinearity = 'prelu'  # special name to separate parameters

from dgl import DGLGraph
import dgl
import tqdm
from torch_geometric.utils.convert import from_scipy_sparse_matrix
if args.dataset in ['citeseer', 'cora']:
    adj_node, features, labels_node, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    features = process.preprocess_features(features)
    origin_adj = adj_node.todense()
    adj_node = sp.csr_matrix(adj_node)
    adj = adj_node
    edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
    features = process.sparse_mx_to_torch_sparse_tensor(features).to_dense()
    idx_test = torch.tensor(idx_test)
    labels = labels_node
    test_lbls = torch.argmax(torch.tensor(labels_node)[idx_test], dim=1)
    if args.model_name == 'FAGCN':
        g = dgl.graph((np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1]))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
    idx_test = idx_test.cuda()
    test_lbls = test_lbls.cuda()
    negative_sample = preprompt.prompt_pretrain_sample(adj, 50)
    neighbors = []
    neighbors_2hops = []
    neighborslist = [[] for x in range(int(origin_adj.shape[0]))]
    neighbors_2hoplist = [[] for x in range(int(origin_adj.shape[0]))]
    neighborsindex = [[] for x in range(int(origin_adj.shape[0]))]
    neighbors_2hopindex = [[] for x in range(int(origin_adj.shape[0]))]
    # neighboradj = adj.todense().A
    for x in tqdm.trange(origin_adj.shape[0]):
        neighborslist[x], neighbors_2hoplist[x] = process.find_2hop_neighbors(origin_adj, x, 20)
        temp1 = [x] * len(neighborslist[x])
        temp2 = [x] * len(neighbors_2hoplist[x])
        # print(temp1)
        neighbors.append(neighborslist[x])
        neighbors_2hops.append(neighbors_2hoplist[x])
        neighborsindex[x] = temp1
        neighbors_2hopindex[x] = temp2
    neighborslist = sum(neighborslist, [])
    neighbors_2hoplist = sum(neighbors_2hoplist, [])
    neighborsindex = sum(neighborsindex, [])
    neighbors_2hopindex = sum(neighbors_2hopindex, [])

    neighbortensor = torch.zeros(len(neighborslist), adj.shape[0])
    neighbors_2hoptensor = torch.zeros(len(neighbors_2hoplist), adj.shape[0])

    for x in tqdm.trange(len(neighborslist)):
        neighbortensor[x][neighborslist[x]] = 1

    for x in tqdm.trange(len(neighbors_2hoplist)):
        neighbors_2hoptensor[x][neighbors_2hoplist[x]] = 1

    nb_nodes = features.shape[0]  # node number
    # ft_size = features.shape[1]  # node features dim
    nb_classes = labels_node.shape[1]  # classes = 6
    if args.model_name == 'FAGCN':
        features = torch.FloatTensor(features).cuda()
    else:
        features = torch.FloatTensor(features[np.newaxis]).cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
if args.dataset in ['PROTEINS','ENZYMES']:
    dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True, )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    data = torch.load(f"data/{args.dataset}/data.pkl")
    features = data.features
    ft_size = features.shape[-1]
    adj = data.adj
    edge_index = torch.tensor(np.array(adj.todense())).nonzero().t().contiguous().cuda()
    if args.model_name == 'FAGCN':
        a = (np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1])
        g = DGLGraph(a)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        g = g.to(edge_index.device)


    origin_adj = np.array(adj.todense())
    labels = data.labels
    edge_index = data.edge_index
    if args.use_origin_feature == 0:
        features = process.preprocess_features(features)
    if args.model_name == 'FAGCN':
        features = torch.FloatTensor(features).cuda()
    else:
        features = torch.FloatTensor(features[np.newaxis]).cuda()


    # test_lbls = torch.load( f"data/fewshot_{args.dataset}/testlabels.pt").type(torch.long).squeeze().cuda()

    adj_ = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj_).cuda()
    if  'graph' not in args.filename:
        if args.dataset == 'PROTEINS':
            path = f"data/fewshot_{args.dataset}/1-shot_{args.dataset}"
        else:
            path = f"data/fewshot_{args.dataset}/5-shot_{args.dataset}"
        test_features = torch.load(f"{path}/testemb.pt")
        if args.use_origin_feature == 0:
            test_features_ = process.preprocess_features(test_features.squeeze(0))
            test_features = torch.FloatTensor(test_features_[np.newaxis]).cuda()
        else:
            test_features = test_features.cuda()
        # test_adj =torch.load(f"data/fewshot_{args.dataset}/testadj.pt").squeeze(0)
        test_adj = torch.load(f"{path}/testadj.pt").squeeze(0)
        if args.model_name =='FAGCN':
            edge_index = test_adj.nonzero().t().contiguous().cuda()
            a = (np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1])
            g_test = DGLGraph(a)
            g_test = dgl.to_simple(g_test)
            g_test = dgl.to_bidirected(g_test)
            g_test = dgl.remove_self_loop(g_test)
            deg = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(deg, -0.5)
            g_test.ndata['d'] = norm
        test_adj_ = test_adj.numpy()
        # test_lbls = torch.load( f"data/fewshot_{args.dataset}/testlabels.pt").type(torch.long).squeeze().cuda()
        test_lbls = torch.load(f"{path}/testlabels.pt").type(
            torch.long).squeeze().cuda()
        adj_csr = sp.csr_matrix(test_adj_)
        test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
        sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()
        # test_lbls = torch.load( f"data/fewshot_{args.dataset}/test_labels.pt").type(torch.long).squeeze().cuda()
        # idx_test = torch.load(f"data/fewshot_{args.dataset}/test_idx.pt").type(torch.long).cuda()
        # # sp_adj_test = test_adj.cuda()
        testneighbors = []
        file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors1shot.csv")
        csvreader = csv.reader(file)
        for row in csvreader:
            testneighbors.append([int(i) for i in row])
        testneighbors_2hop = []
        file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors_2hop1shot.csv")
        csvreader = csv.reader(file)
        for row in csvreader:
            testneighbors_2hop.append([int(i) for i in row])
    else:
        path = f"data/fewshot_{args.dataset}_graph"

        test_features = torch.load(f"{path}/testset/feature.pt")
        if args.use_origin_feature == 0:
            test_features = process.preprocess_features(test_features)
        if args.model_name == 'FAGCN':
            test_features = torch.FloatTensor(test_features).cuda()
        else:
            test_features = torch.FloatTensor(test_features[np.newaxis]).cuda()
        test_adj = torch.load(f"{path}/testset/adj.pt")
        if args.model_name =='FAGCN':
            edge_index = test_adj.nonzero().t().contiguous().cuda()
            a = (np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1])
            g_test = DGLGraph(a)
            g_test = dgl.to_simple(g_test)
            g_test = dgl.to_bidirected(g_test)
            g_test = dgl.remove_self_loop(g_test)
            deg = g_test.in_degrees().float().clamp(min=1)
            norm = torch.pow(deg, -0.5)
            g_test.ndata['d'] = norm
        test_adj_ = test_adj.numpy()
        adj_csr = sp.csr_matrix(test_adj_)
        test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
        sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()
        test_lbls = torch.load(f"{path}/testset/labels.pt")
        nb_classes = len(torch.unique(test_lbls))
        test_graph_len = torch.load(f"{path}/testset/graph_len.pt")
        if not os.path.exists(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/testneighbors1_graph.csv"):
            testneighbors = [[] for m in range(len(test_adj_))]
            testneighbors_2hop = [[] for m in range(len(test_adj_))]
            for step, x in enumerate(range(test_adj_.shape[0])):
                tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(test_adj_, x)
                testneighbors[step] = tempneighbors
                testneighbors_2hop[step] = tempneighbors_2hop
            with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/testneighbors1_graph.csv",
                      "w") as f:
                wr = csv.writer(f)
                wr.writerows(testneighbors)
            with open(
                    f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/testneighbors_2hop_graph.csv",
                    "w") as f:
                wr = csv.writer(f)
                wr.writerows(testneighbors_2hop)
        else:
            testneighbors = []
            file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/testneighbors1_graph.csv")
            csvreader = csv.reader(file)
            for row in csvreader:
                testneighbors.append([int(i) for i in row])
            testneighbors_2hop = []
            file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/testneighbors_2hop_graph.csv")
            csvreader = csv.reader(file)
            for row in csvreader:
                testneighbors_2hop.append([int(i) for i in row])

    nb_classes = len(torch.unique(test_lbls))

# test_adj = torch.load(f"data/fewshot_{args.dataset}/5-shot_{args.dataset}/testadj.pt").squeeze(0)
# test_adj_ = test_adj.numpy()
# adj_csr = sp.csr_matrix(test_adj_)
# test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
# sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()

#
# a1 = 0.9    #dgi
# a2 = 0.9    #graphcl
# a3 = 0.1    #lp
# ft_size = 1
#
# device = torch.device("cuda")
# model = PrePrompt(ft_size, hid_units, nonlinearity,a1,a2,a3,1,0.3)
# model = model.to(device)

sparse = True
print("")

lista4=[0.0001]

best_accs=0

from graphcl.dgi import DGI


            # model = DGI(ft_size, hid_units, 'prelu')
model = PrePrompt(ft_size, args.hid_units,args.num_layers,0.05,
                  model_name=args.model_name, reduction=args.reduction,hop_level=args.hop_level, gp=args.gp,weight=args.weight,
                  concat_dense=args.concat_dense, g=g if args.model_name == 'FAGCN' else None)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)
import tqdm
if not os.path.exists(args.save_name):
    for epoch in range(nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        loss = 0
        regloss = 0
        if args.dataset in ['cora','citeseer']:
            b_xent = nn.BCEWithLogitsLoss()
            xent = nn.CrossEntropyLoss()
            cnt_wait = 0
            best = 1e9
            best_t = 0

            model.train()
            optimiser.zero_grad()
            if args.model_name == 'GCN':
                logit = model(features, adj=sp_adj, LP=LP, sample=torch.tensor(negative_sample).cuda(),
                              neighborslist=neighbortensor.cuda(), neighbors_2hoplist=neighbors_2hoptensor.cuda(),
                              neighborsindex=torch.tensor(neighborsindex).cuda(),
                              neighbors_2hopindex=torch.tensor(neighbors_2hopindex).cuda())
            if args.model_name == 'FAGCN':
                logit = model(features, LP=LP, g=g_train, sample=torch.tensor(negative_sample).cuda(),
                              neighborslist=neighbortensor.cuda(), neighbors_2hoplist=neighbors_2hoptensor.cuda(),
                              neighborsindex=torch.tensor(neighborsindex).cuda(),
                              neighbors_2hopindex=torch.tensor(neighbors_2hopindex).cuda())
            loss += logit
        else:
            for step, data in enumerate(loader):
                # print(step)
                features,adj,nodelabels= process.process_tu(data,ft_size)
                adj_ = adj.todense()
                if args.model_name == 'GCN':
                    features = torch.FloatTensor(features[np.newaxis]).cuda()
                if args.model_name == 'FAGCN':
                    features = torch.FloatTensor(features).cuda()
                    edge_index = torch.tensor(adj_).nonzero().t().contiguous().cuda()
                    a = (np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1])
                    g_train = DGLGraph(a)
                    g_train = dgl.to_simple(g_train)
                    g_train = dgl.to_bidirected(g_train)
                    g_train = dgl.remove_self_loop(g_train)
                    deg = g_train.in_degrees().float().clamp(min=1)
                    norm = torch.pow(deg, -0.5)
                    g_train.ndata['d'] = norm
                    g_train = g_train.to(features.device)
                origin_adj = np.array(adj_)
                negative_sample = preprompt.prompt_pretrain_sample(adj,50)
                neighbors = []
                neighbors_2hops = []
                neighborslist = [[] for x in range(int(origin_adj.shape[0]))]
                neighbors_2hoplist = [[] for x in range(int(origin_adj.shape[0]))]
                neighborsindex = [[] for x in range(int(origin_adj.shape[0]))]
                neighbors_2hopindex = [[] for x in range(int(origin_adj.shape[0]))]
                # neighboradj = adj.todense().A
                for x in tqdm.trange(origin_adj.shape[0]):
                    neighborslist[x], neighbors_2hoplist[x] = process.find_2hop_neighbors(origin_adj, x, 20)
                    temp1 = [x] * len(neighborslist[x])
                    temp2 = [x] * len(neighbors_2hoplist[x])
                    # print(temp1)
                    neighbors.append(neighborslist[x])
                    neighbors_2hops.append(neighbors_2hoplist[x])
                    neighborsindex[x] = temp1
                    neighbors_2hopindex[x] = temp2
                neighborslist = sum(neighborslist,[])
                neighbors_2hoplist = sum(neighbors_2hoplist,[])
                neighborsindex = sum(neighborsindex,[])
                neighbors_2hopindex = sum(neighbors_2hopindex,[])

                neighbortensor = torch.zeros(len(neighborslist),adj.shape[0])
                neighbors_2hoptensor = torch.zeros(len(neighbors_2hoplist),adj.shape[0])

                for x in tqdm.trange(len(neighborslist)):
                    neighbortensor[x][neighborslist[x]] = 1

                for x in tqdm.trange(len(neighbors_2hoplist)):
                    neighbors_2hoptensor[x][neighbors_2hoplist[x]] = 1

                nb_nodes = features.shape[0]  # node number
                # ft_size = features.shape[1]  # node features dim
                nb_classes = nodelabels.shape[1]  # classes = 6


                adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
                sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)


                if torch.cuda.is_available():
                    print('Using CUDA')
                    model = model.cuda()
                    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
                    features = features.cuda()
                    if sparse:
                        sp_adj = sp_adj.cuda()
                # labels = labels.cuda()
                b_xent = nn.BCEWithLogitsLoss()
                xent = nn.CrossEntropyLoss()
                cnt_wait = 0
                best = 1e9
                best_t = 0

                model.train()
                optimiser.zero_grad()
                if args.model_name =='GCN' :
                    logit = model(features, adj= sp_adj,LP=LP,sample=torch.tensor(negative_sample).cuda(),neighborslist=neighbortensor.cuda(),neighbors_2hoplist=neighbors_2hoptensor.cuda(),
                                  neighborsindex=torch.tensor(neighborsindex).cuda(),neighbors_2hopindex=torch.tensor(neighbors_2hopindex).cuda())
                if args.model_name =='FAGCN' :
                    logit = model(features,LP=LP,g=g_train,sample=torch.tensor(negative_sample).cuda(),neighborslist=neighbortensor.cuda(),neighbors_2hoplist=neighbors_2hoptensor.cuda(),
                                  neighborsindex=torch.tensor(neighborsindex).cuda(),neighbors_2hopindex=torch.tensor(neighbors_2hopindex).cuda())
                # print('Loss:[{:.4f}]'.format(loss.item()))
                loss = loss + logit

            loss = loss / step
            print('Loss:[{:.4f}]'.format(loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(args.save_name))
model.eval()
model.cuda()

tot = torch.zeros(1)
tot = tot.cuda()
accs = []
# train_range = range(1701)
print('-' * 100)
cnt_wait = 0
best = 1e9
best_t = 0

xent = nn.CrossEntropyLoss()
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum",shotnum)
    for i in range(100):
        if args.dataset in ['cora','citeseer']:
            model = model.cuda()
            test_embs = torch.index_select(embeds, 0, torch.tensor(list(idx_test)).cuda())
            # preval_embs = embeds[0, idx_val]
            # test_embs = embeds[0, idx_test]
            # val_lbls = torch.argmax(labels[0, idx_val], dim=1)
            test_adj = adj.todense().A
            # test_adj = test_adj[:,idx_test][idx_test,:].A
            testneighbors = [[] for m in range(len(idx_test))]
            testneighbors_2hop = [[] for m in range(len(idx_test))]
            for step, x in enumerate(idx_test):
                tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(origin_adj, idx_test[step])
                testneighbors[step] = tempneighbors
                testneighbors_2hop[step] = tempneighbors_2hop

        else:
            adj_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/adj.pt")
            if args.model_name == 'FAGCN':
                edge_index = adj_train.nonzero().t().contiguous().cuda()
                a = (np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1])
                g_train = DGLGraph(a)
                g_train = dgl.to_simple(g_train)
                g_train = dgl.to_bidirected(g_train)
                g_train = dgl.remove_self_loop(g_train)
                deg = g_train.in_degrees().float().clamp(min=1)
                norm = torch.pow(deg, -0.5)
                g_train.ndata['d'] = norm
                g_train = g_train.to(features.device)
            feature_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/feature.pt")
            train_lbls = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(torch.long).squeeze().cuda()
            train_graph_len = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/graph_len.pt")
            if args.use_origin_feature == 0:
                feature_train = process.preprocess_features(feature_train)
            feature_train = torch.FloatTensor(feature_train[np.newaxis]).cuda()

            adj_train_ = adj_train.numpy()
            adj_csr = sp.csr_matrix(adj_train_)
            adj_train = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
            sp_adj_train = process.sparse_mx_to_torch_sparse_tensor(adj_train).cuda()

            neighbors = [[] for m in range(len(adj_train_))]
            neighbors_2hop = [[] for m in range(len(adj_train_))]
            for step, x in enumerate(range(len(adj_train_))):
                tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(adj_train_, x)
                neighbors[step] = tempneighbors
                neighbors_2hop[step] = tempneighbors_2hop
            if args.model_name == 'FAGCN':
                embeds, _ = model.embed(feature_train, sp_adj_train, sparse, None,LP,g=g_train)

            else:
                embeds, _ = model.embed(feature_train, sp_adj_train, sparse, None,LP)
            if embeds.dim() == 3:
                embeds = embeds.squeeze(0)
            print("true",i,train_lbls)

        log = downprompt(neighbors, neighbors_2hop, nb_classes, embeds, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction, multi_prompt=args.multi_prompt,
                         concat_dense=None, bottleneck_size=args.bottleneck_size,
                         meta_in=args.meta_in, out_size=args.out_size,
                         hidden_size=args.hid_units, prompt=args.prompt, use_metanet=args.use_metanet)
        # log = downprompt(neighbors,neighbors_2hop, hid_units, nb_classes,embeds,train_lbls,model.tokens.weight[0][0],model.tokens.weight[0][1],model.tokens.weight[0][2])
        # log = downprompt(neighbors,neighbors_2hop, hid_units, nb_classes,embeds,train_lbls,hop_level=args.hop_level,reduction=args.reduction,
        #                  q=model.attention.query,k=model.attention.key,v=model.attention.value,multi_prompt=args.multi_prompt)
        # q=model.attention.query,k=model.attention.key,v=model.attention.value)
        # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
        # opt = torch.optim.Adam([*log.parameters(),*feature_prompt.parameters()], lr=0.001)
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
            logits = log.forward2(train=1, train_embeds=embeds, graph_len=train_graph_len).float().cuda()
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
            test_embs, _ = model.embed(test_features, sp_adj_test, sparse, None, LP,g=g_test if args.model_name =='FAGCN' else None)

            # logits = log(test_embs,neighbors=testneighbors,neighbors_2hop=testneighbors_2hop)#train_adj=test_adj)
            logits = log.forward2(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,
                                  test_embeds=test_embs, graph_len=test_graph_len)
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
        zip([args.model_name], [args.save_name], [args.hop_level], [args.aug_type],
            [args.multi_prompt], [args.prompt], [args.use_metanet], [args.bottleneck_size],
            [accs.mean().item()], [accs.std().item()], [shotnum], [args.use_origin_feature])),
        columns=['model_name', 'pretrained_model', 'hop_level', 'aug_type', 'multi_prompt',
                 'prompt', 'metanet', 'bottleneck', 'accuracy', 'mean', 'shotnum', 'use_origin_feature'])
    if os.path.exists(f"data/{args.dataset}/{args.filename}.csv"):
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv", mode='a', header=False)
    else:
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv")