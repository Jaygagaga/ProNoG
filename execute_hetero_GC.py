import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import scipy.sparse as sp

import random

from preprompt_new1 import PrePrompt

import dgl


from utils import process



import argparse
from downprompt_metanet3 import downprompt
import csv
import torch_geometric
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset

import tqdm
# from models.H2GCN.utils import  eidx_to_sp

# from utils.heterophilic import WebKB, WikipediaNetwork, Actor

from torch_geometric.loader import DataLoader
parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="chameleon", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.05, help='drop percent')
parser.add_argument('--drop_edge', type=float, default=0.2, help='drop percent')
parser.add_argument('--dropout', type=float, default=0, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
# parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--use_origin_feature',default=True,   action="store_true", help='aug type: mask or edge')
parser.add_argument('--nb_epochs', type=int, default=5000, help='number of epochs')
# pretraining hyperperemeter
parser.add_argument('--pretrained_model', type=str, default='/home/xingtong/WebKB_node_origin/modelset/graphcl/FAGCN_graphCL_chameleon_70.pth',
                    help='save ckpt name')
parser.add_argument('--num_layers', type=int, default=2, help='num of layers')
parser.add_argument('--model_name', type=str, default='FAGCN', help='save ckpt name')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--down_lr', type=float, default=0.0001, help='lr for downstream')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=50, help='patience')

# downstream hyperperemeter
parser.add_argument('--prompt', type=int, default=0, help='0:no prompt,1:use prompt')
parser.add_argument('--multi_prompt', type=int, default=0, help='1:multi prompt or 0:single prompt')
parser.add_argument('--use_metanet', type=int, default=1, help='use metanet layer or not')
parser.add_argument('--meta_in', type=int, default=256, help='hidden size of metanet input')
parser.add_argument('--bottleneck_size', type=int, default=64, help='bottleneck size')
parser.add_argument('--out_size', type=int, default=256, help='hidden size of metanet output')
parser.add_argument('--gp', type=int, default=1, help='0: no subgraph, 1: subgraph')
parser.add_argument('--weight', type=int, default=1, help='0: no weights, 1: weights')
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--down_weight', type=int, default=0, help='0: no downstream weights, 1: weights')
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--sample_num', type=int, default=70, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=256, help='number of neighbors')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='no_prompt_with_metanet_graph_origin_feature2', help='filename')




args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)



seed = args.seed
random.seed(seed)
np.random.seed(seed)
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from models.FAGCN.utils import normalize_features

sparse = True
useMLP = False
import torch_geometric.transforms as T
nonlinearity = 'prelu'  # special name to separate parameters
import utils.aug as aug
from utils.heterophilic import  WikipediaNetwork, Actor
# if args.dataset =='cornell':
if args.dataset in ['cornell', 'texas', 'wisconsin']:
    datasets = WebKB(root='data', name=args.dataset)
    # features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # idx_test = range(features.shape[0]-100,features.shape[0])
    # origin_adj = adj.todense().A

if args.dataset in ['chameleon', 'squirrel']:
    datasets = WikipediaNetwork(root='data', name=args.dataset)
features, node_adj, labels, _, _ = process.process_webkb(datasets.data, datasets.data.x.shape[0])
# if args.dataset in ['film']:
#     datasets = Actor(root='data')
# origin_datasets = torch.load(f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data.pkl')

# num_nodes_origin = [i.num_nodes for i in origin_datasets]
# idx = [[j] * i for i, j in zip(num_nodes_origin, range(len(num_nodes_origin)))]
# graph_datasets= torch.load(f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data_augmented_{args.sample_num}_{args.aug_type}_NC.pkl')
# graph_datasets= torch.load(f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/graph_data_augmented_{args.sample_num}_{args.aug_type}.pkl')
#
#
# loader = DataLoader(graph_datasets,batch_size=(args.sample_num+2)*args.batch_size,follow_batch=['x']*(args.sample_num+2)*args.batch_size,shuffle=False)

def get_feature_and_g(graph_datasets):
    graph_list = []
    graph_labels = []

    for num, i in enumerate(graph_datasets):
        data = dgl.graph((np.array(i.edge_index)[0], np.array(i.edge_index)[1]))
        data.ndata['x'] = i.x
        graph_labels.append(i.y.item())
        graph_list.append(data)
    # concated graph and adj for the origin graph dataset (not including positive and negative graph samples)
    g_origin = dgl.batch(graph_list)
    # adj_origin = torch_geometric.utils.to_dense_adj(torch.stack(g_origin.edges(), 1))
    # adj_origin = adj_origin.squeeze(0)
    # adj_origin = sp.csr_matrix(adj_origin.numpy())
    # adj_origin = process.normalize_adj(adj_origin + sp.eye(adj_origin.shape[0]))
    features_origin = g_origin.ndata['x'].cuda()
    # if sparse:
    #     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    if args.model_name == 'FAGCN':
        if args.use_origin_feature == False:
            features_origin = normalize_features(g_origin.ndata['x'])
            features_origin =  torch.FloatTensor(features_origin).cuda()
        g_origin = dgl.to_simple(g_origin)
        g_origin = dgl.remove_self_loop(g_origin)
        g_origin = dgl.to_bidirected(g_origin)

            # features_origin = g_origin.ndata['x'].cuda()
        g_origin = g_origin.to(features_origin.device)
        deg = g_origin.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g_origin.ndata['d'] = norm
    graph_labels = torch.tensor(graph_labels).cuda()
    return features_origin, g_origin,graph_labels


idx_test = torch.load(f"data/fewshot_{args.dataset}_graph/test_idx.pt").type(torch.long).cuda()
test_lbls = torch.load(f"data/fewshot_{args.dataset}_graph/test_labels.pt").type(torch.long)
if args.dataset == 'cornell':
    origin_adj = node_adj.todense().A.T
else:
    origin_adj = node_adj.todense().A

# features_origin, g_origin,graph_labels = get_feature_and_g(origin_datasets)
if args.model_name =='FAGCN':
    from models.FAGCN.utils import preprocess_data
    g_origin, nclass, features_origin, labels_origin = preprocess_data(args.dataset, 1)
    # features_origin = features #torch.where(features_origin!=0.0,1.0,0.0)
    features_origin = features_origin.cuda()
    labels_origin = labels_origin.cuda()

    g_origin = g_origin.to(features_origin.device)
    deg = g_origin.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g_origin.ndata['d'] = norm
if args.use_origin_feature ==True:
    features_origin = features.cuda()
else:
    features_origin = features_origin.cuda()
nb_nodes = features_origin.shape[0]  # node number
ft_size = features_origin.shape[-1]  # node features dim
nb_classes = labels.shape[-1]  # classes = 6
device="cuda"

LP = False
model = PrePrompt(ft_size, args.hid_units,args.num_layers,args.drop_percent,
                  model_name=args.model_name, reduction=args.reduction,hop_level=args.hop_level, gp=args.gp,weight=args.weight,
                  eps=args.eps,aug_type = args.aug_type, sample_num = args.sample_num)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if torch.cuda.is_available():
    print('Using CUDA')
    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    # features_origin = features_origin.cuda()
    # if sparse:
    #     sp_adj = sp_adj.cuda()
    # else:
    #     adj = adj.cuda()
    # graph_labels = graph_labels.cuda()

'''
------------------------------------------------------------
Pretraining
------------------------------------------------------------
'''
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

best = 1e9
best_t = 0
cnt_wait = 0
# if not os.path.exists(args.pretrained_model):
#     for epoch in range(args.nb_epochs):
#         if cnt_wait >= args.patience:
#             print('Early stopping!')
#             break
#         for batch in loader:
#             model.train()
#             optimiser.zero_grad()
#
#             feature = batch.x.cuda()
#             edge_index = batch.edge_index.cuda()
#             inputs_indices = batch.batch
#             adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
#             adj = sp.csr_matrix(adj.cpu().numpy())
#             adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
#             if sparse:
#                 sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
#             g= dgl.graph((np.array(batch.edge_index)[0], np.array(batch.edge_index)[1]))
#             g = dgl.to_simple(g)
#             g = dgl.remove_self_loop(g)
#             g = dgl.to_bidirected(g)
#             g = g.to(feature.device)
#             deg = g.in_degrees().cuda().float().clamp(min=1)
#             norm = torch.pow(deg, -0.5)
#             g.ndata['d'] = norm
#
#             if args.model_name =='FAGCN':
#                 loss= model(feature=feature,temperature=10,inputs_indices=inputs_indices,g=g) # feature, shuf_fts,aug_features1,aug_features2,sp_adj,sp_aug_adj1,sp_aug_adj2,sparse,aug_type):
# #             elif args.model_name =='H2GCN':
# #                 loss = model(feature=negative_fea_tensor,adj=h2gcn_adj, temperature=10)
# #             elif args.model_name =='DGI':
# #                 loss = model(feature=features, temperature=10)
#
#             if loss < best:
#                 best = loss
#                 best_t = epoch
#                 cnt_wait = 0
#                 torch.save(model.state_dict(), args.pretrained_model)
#             else:
#                 cnt_wait += 1
#             # if cnt_wait >= args.patience:
#             #     print('Early stopping!')
#             #     break
#             loss.backward()
#             optimiser.step()
#
#         print('epoch: {}--Loss:[{:.4f}]'.format(epoch,loss.item()))
#     print('Loading {}th epoch {}'.format(best_t, loss.item()))
try:
    model.load_state_dict(torch.load(args.pretrained_model))
except:
    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
embeds, _ = model.embed(feature=features_origin, g=g_origin)

neighbors = []
file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/neighbors.csv")
csvreader = csv.reader(file)
for row in csvreader:
    neighbors.append([int(i) for i in row])
neighbors_2hops = []
file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/neighbors_2hops.csv")
csvreader = csv.reader(file)
for row in csvreader:
    neighbors_2hops.append([int(i) for i in row])


origin_datasets = torch.load(f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data1.pkl')
test_datasets = [origin_datasets[t] for t in idx_test]
for n in tqdm.trange(len(test_datasets)):#get neighbors for each node in each graph (test_datasets is a list of graphs)
    nodes = test_datasets[n].graph_nodes[0]
    test_neighors = [neighbors[i] for i in nodes]
    test_datasets[n].test_neighors = test_neighors
    test_neighbors_2hops = [neighbors_2hops[i] for i in nodes]
    test_datasets[n].test_neighors_2hop = test_neighbors_2hops
    # test_embs = torch.index_select(embeds, 0, torch.tensor(nodes).cuda())
    # test_datasets[n].test_embs =test_embs
    # num_nodes_test = [len(i) for i in test_datasets[n].graph_nodes]
    # idx = [[j] * i for i, j in zip(num_nodes_test, range(len(num_nodes_test)))]
    # test_datasets[n].idx = idx
test_loader = DataLoader(test_datasets, batch_size=len(test_datasets), follow_batch=['x'] * len(test_datasets),shuffle=False)

'''
------------------------------------------------------------
Downstream prompting
------------------------------------------------------------
'''
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum",shotnum)
    for i in range(100):
        idx_train = torch.load(f"data/fewshot_{args.dataset}_graph/{shotnum}-shot_{args.dataset}/{i}/idx.pt").type(torch.long).cuda()
        train_lbls = torch.load(f"data/fewshot_{args.dataset}_graph/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(torch.long).squeeze().cuda()

        print(f'NO.{i} {idx_train}')
        # print(len(idx_train))
        train_datasets = [origin_datasets[t] for t in idx_train]
        train_graph_nodes = [i.graph_nodes[0] for i in train_datasets]
        train_graph_nodes = sum(train_graph_nodes, [])
        train_embs = torch.index_select(embeds, 0, torch.tensor(train_graph_nodes).cuda())
        features_origin, g_origin, graph_labels = get_feature_and_g(train_datasets)
        adj_ = torch_geometric.utils.to_dense_adj(torch.stack(g_origin.edges(), 0)).squeeze(0).cpu().numpy()
        neighbors = [[]for m in range(len(adj_)) ]
        neighbors_2hop = [[]for m in range(len(adj_))  ]
        # print(neighbors)
        num_nodes_train = [i.num_nodes for i in train_datasets]
        idx = [[j]*i for i,j in zip(num_nodes_train, range(len(num_nodes_train)))]
        for n in range(len(adj_)):
            tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(adj_, n, args.k)
            # print(n, tempneighbors, tempneighbors_2hop)
            neighbors[n] = tempneighbors
            neighbors_2hop[n] = tempneighbors_2hop

#         #downstream prompting
        log = downprompt( neighbors,neighbors_2hop,nb_classes, train_embs, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction,multi_prompt=args.multi_prompt,
                         bottleneck_size=args.bottleneck_size,meta_in=args.meta_in, dropout=args.dropout, out_size = args.out_size,
                         activation = args.activation, hidden_size=args.hid_units,prompt=args.prompt,use_metanet=args.use_metanet)

        opt = torch.optim.Adam(log.parameters(), lr=args.down_lr)
        log.cuda()
        best = 1e9
        pat_steps = 0
        cnt_wait = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        best_t = 0
        for _ in range(200):
            log.train()
            opt.zero_grad()

            logits = log.forward_graph(train=1,idx=idx).float().cuda()
            loss = xent(logits, train_lbls)
            if not loss.requires_grad:
                loss.requires_grad = True

            if loss < best:
                best = loss
                best_t = _
                cnt_wait = 0
                # torch.save(model.state_dict(), args.pretrained_model)
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            loss.backward(retain_graph=True)
            opt.step()
        print('{}--{}'.format(best_t,loss))
        preds = []
        for num, batch in enumerate(test_loader):
            print(num)
            num_nodes_test = [len(i[0]) for i in batch.graph_nodes]
            idx = [[j] * i for i, j in zip(num_nodes_test, range(len(num_nodes_test)))]
            # idx = sum(idx,[])
            # idx=torch.tensor(idx).cuda()
            # # embeds, _ = model.embed(feature=features_origin_test, g=g_origin_test)
            neighbors_test = batch.test_neighors
            neighbors_test=sum(neighbors_test,[])
            neighbors_2hop_test = batch.test_neighors_2hop
            neighbors_2hop_test = sum(neighbors_2hop_test, [])
            nodes = [i[0] for i in batch.graph_nodes]
            nodes= sum(nodes,[])
            # test_embs = embeds[nodes]
            test_embs = torch.index_select(embeds, 0, torch.tensor(nodes).cuda())
            # idx = batch.idx.cuda()
            logits = log.forward_graph(train=0, neighbors=neighbors_test, neighbors_2hop=neighbors_2hop_test,idx=idx,test_embeds=test_embs,embeds=embeds)
            pred = torch.argmax(logits, dim=1)
            preds += pred.tolist()
        acc = torch.sum(torch.tensor(preds)== test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print('acc:[{:.4f}]'.format(acc))
        tot += acc
        # break

    print('-' * 100)
    # print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
    # accs = torch.stack(accs)
    try:
        accs = torch.tensor(accs)
    except:
        accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)
    df = pd.DataFrame(list(zip([args.model_name],[args.pretrained_model],[args.num_layers], [args.weight],[args.lr],[args.down_lr],[args.weight_decay],[args.reduction],[args.hop_level],[args.gp],[args.drop_percent],[args.dropout],[args.drop_edge], [args.aug_type],[args.prompt], [args.multi_prompt],[args.use_metanet],[args.k],
                               [args.sample_num],[args.hid_units],[args.bottleneck_size],[args.meta_in],[accs.mean().item()],[accs.std().item()],[shotnum])),
    columns= ['model_name' ,'pretrained_model','num_layers', 'weight', 'lr','down_lr','weight_decay','reduction', 'hop level','gp','drop_percent','dropout','drop_edge','aug_type','prompt','multi_prompt','use_metanet','k', 'sample_num', 'hid_units', 'bottleneck_size','meta_in', 'accuracy', 'mean', 'shotnum'])
    if os.path.exists(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv"):
        df.to_csv(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv",mode='a',header=False)
    else:
        df.to_csv(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv")

