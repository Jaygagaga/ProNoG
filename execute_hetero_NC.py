import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import scipy.sparse as sp

import random
from preprompt_new1 import PrePrompt
import dgl
from dgl import DGLGraph
from utils import process
from torch_geometric.datasets import HeterophilousGraphDataset
from models.FAGCN.utils import accuracy

import argparse
from downprompt_metanet3 import downprompt
import csv
import torch_geometric
from torch_geometric.datasets import WebKB
import tqdm
from models.H2GCN.utils import  eidx_to_sp

# from utils.heterophilic import WebKB, WikipediaNetwork, Actor

from torch_geometric.loader import DataLoader
parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="cornell", help='data')
parser.add_argument('--aug_type', type=str, default="mask", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.05, help='drop percent')
parser.add_argument('--drop_edge', type=float, default=0.2, help='drop percent for edge dropping')
parser.add_argument('--dropout', type=float, default=0, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
# parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--use_origin_feature',default=False,   action="store_true", help='aug type: mask or edge')

# pretraining hyperperemeter
parser.add_argument('--pretrained_model', type=str, default='modelset/graphcl/FAGCN_graphCL_cornell_origin_feature.pkl',
                    help='save ckpt name')
parser.add_argument('--num_layers', type=int, default=2, help='num of layers')
parser.add_argument('--model_name', type=str, default='FAGCN', help='save ckpt name')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--down_lr', type=float, default=0.0001, help='lr for downstream')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=50, help='patience')
parser.add_argument('--nb_epochs', type=int, default=2000, help='number of epochs')
parser.add_argument('--sample_num', type=int, default=70, help='number of negative samples')
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
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--hid_units', type=int, default=256, help='hidden size')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='FAGCN_no_prompt_with_metanet_origin_feature2', help='number of neighbors')


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

# nb_epochs = 5000
patience = 50


sparse = True



import utils.aug as aug
from utils.heterophilic import  WikipediaNetwork, Actor


if args.dataset in ['cornell', 'texas', 'wisconsin']:
    datasets = WebKB(root='data', name=args.dataset)
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # idx_test = range(features.shape[0]-100,features.shape[0])
    origin_adj = adj.todense().A
    if args.dataset == 'cornell':
        origin_adj = adj.todense().A.T
if args.dataset in ['chameleon', 'squirrel']:
    datasets = WikipediaNetwork(root='data', name=args.dataset)
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    origin_adj = adj.todense().A
# if args.dataset in ['film']:
#     datasets = Actor(root='data')

# features= process.preprocess_features(features)
# features = torch.FloatTensor(features[np.newaxis])
# features = features.cuda()
labels = torch.FloatTensor(labels[np.newaxis]).cuda()
idx_test = torch.load(f"data/fewshot_{args.dataset}/test_idx.pt").type(torch.long).cuda()
test_lbls = torch.load(f"data/fewshot_{args.dataset}/test_labels.pt").type(torch.long).cuda()


from torch_geometric.utils.convert import from_scipy_sparse_matrix
edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)

# print("labels",labels)
print("adj",sp_adj.shape)
print("feature",features.shape)
# negetive_sample = preprompt_new.prompt_pretrain_sample(adj,args.sample_num)
# negetive_sample =preprompt_metanet.prompt_pretrain_sample(adj,70)
nb_nodes = features.shape[0]  # node number
ft_size = features.shape[-1]  # node features dim
nb_classes = labels.shape[-1]  # classes = 6
device="cuda"
from models.FAGCN.utils import preprocess_data
if args.model_name =='FAGCN':

    if args.dataset in ['cornell', 'texas', 'wisconsin','chameleon', 'squirrel']:
        g_origin, nclass, features_origin, labels_origin = preprocess_data(args.dataset, 1)
        features_origin = features_origin.cuda()
        labels_origin = labels_origin.cuda()

        g_origin = g_origin.to(features_origin.device)
        deg = g_origin.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g_origin.ndata['d'] = norm
if args.model_name =='H2GCN':
    from models.H2GCN.utils import load_dataset
    features_origin, _, _, _, adj_origin= load_dataset(args.dataset,device)
    features_origin = features_origin.cuda()
    adj_origin = adj_origin.cuda()
    # features_origin=features
if args.model_name =='H2GCN':
    adj_origin = sp_adj

if args.use_origin_feature ==True:
    features_origin = features.cuda()
else:
    features_origin = features_origin.cuda()
# else:
#     features_origin = torch.FloatTensor(features[np.newaxis])

#
# features_origin = torch.FloatTensor(features_origin[np.newaxis])
"""
------------------------------------------------------------
2hop neighbors and subgraphs
------------------------------------------------------------
"""
if not os.path.exists(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors_2hops.csv"):
    neighbors = []
    neighbors_2hops = []
    neighborslist = [[] for x in range(origin_adj.shape[0])]
    neighbors_2hoplist = [[] for x in range(origin_adj.shape[0])]
    neighborsindex = [[] for x in range(origin_adj.shape[0])]
    neighbors_2hopindex = [[] for x in range(origin_adj.shape[0])]
    # neighboradj = adj.todense().A
    for x in tqdm.trange(origin_adj.shape[0]):
        neighborslist[x], neighbors_2hoplist[x]= process.find_2hop_neighbors(origin_adj,x,args.k)
        temp1 = [x] *len(neighborslist[x])
        temp2 = [x] *len(neighbors_2hoplist[x])
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

    with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(neighbors)
    with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors_2hops.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(neighbors_2hops)
else:
    neighbors = []
    file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors.csv")
    csvreader = csv.reader(file)
    for row in csvreader:
        neighbors.append([int(i) for i in row])
    neighbors_2hops = []
    file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors_2hops.csv")
    csvreader = csv.reader(file)
    for row in csvreader:
        neighbors_2hops.append([int(i) for i in row])


# neighborsindex = neighborsindex.cusa()



'''
------------------------------------------------------------
edge / mask subgraph
------------------------------------------------------------
'''
if not os.path.exists(args.pretrained_model) :

    if args.model_name != "DGI":
        if not os.path.exists(f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_augmented_negative_indices.txt'):
            if not os.path.exists(f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_merged_edges.pkl'):
                print('Starting augmentation-------------')
                """Get extended adjancy and graph"""
                subgraphs = []
                subfeatures = []
                subadjs = []
                count_nodes = 0
                merged_edges = []
                for node in tqdm.trange(nb_nodes):
                    subset_nodes = [node] + neighbors[node] + neighbors_2hops[node]
                    if len(subset_nodes) > 1:
                        new_edge_index, _ = torch_geometric.utils.subgraph(edge_index=edge_index.cpu(), subset=subset_nodes,
                                                                           relabel_nodes=True)
                        # dict = {i: i + count_nodes + 1 for i in np.unique(new_edge_index)}
                        # row = [dict[i] for i in new_edge_index[0].tolist()]
                        # col = [dict[i] for i in new_edge_index[1].tolist()]
                        dic = {i: i + count_nodes + 1 for i in np.unique(new_edge_index)}
                        row = [dic[i] - 1 for i in new_edge_index[0].tolist()]
                        col = [dic[i] - 1 for i in new_edge_index[1].tolist()]
                        new_edge_index_ = torch.stack((torch.tensor(row), torch.tensor(col)), 0)
                        merged_edges.append(new_edge_index_)
                        new_adj = torch_geometric.utils.to_dense_adj(new_edge_index_).squeeze(0)

                        subadjs.append(sp.csr_matrix(new_adj))
                        subset_nodes_new = np.unique(new_edge_index_)
                    # else:
                    #     if count_nodes !=0:
                    #         subset_nodes_new = [count_nodes]
                    #     else:
                    #         subset_nodes_new = [0]
                    #     subadjs.append(None)
                        subgraphs.append(list(subset_nodes_new))

                        count_nodes += len(subset_nodes)
                        subfeatures.append(features[np.unique(subset_nodes)]) #Must use original 0/1 features
                merged_edges = torch.cat(merged_edges, dim=1)
                subfeatures = torch.cat(subfeatures, dim=0)
                torch.save(merged_edges,
                           f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_merged_edges.pkl')
                torch.save(subfeatures,
                           f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subfeatures.pkl')
                with open(f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subgraphs.csv', "w") as f:
                    wr = csv.writer(f)
                    wr.writerows(subgraphs)

            else:
                merged_edges = torch.load( f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_merged_edges.pkl')
                subfeatures = torch.load(f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subfeatures.pkl')
                subgraphs = []
                file = open(
                    f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subgraphs.csv')
                csvreader = csv.reader(file)
                for row in csvreader:
                    subgraphs.append([int(i) for i in row])
            print("--get negative subgraph samples")

            negative_fea_tensor, merged_edges_, negative_indices = process.get_negative_subgraphs(args.sample_num, subfeatures,subgraphs,
                                                                                                  merged_edges,
                                                                                                  aug_type=args.aug_type,
                                                                                                  drop_edge=0.2)
            from torch_geometric.data import Data
            data = Data(x=negative_fea_tensor, edge_index=merged_edges_)
            torch.save(data, f'data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_augmented_data.pkl')
            with open(f'data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_augmented_negative_indices.txt', 'w') as fp:
                for item in negative_indices:
                    # write each item on a new line
                    fp.write("%s\n" % str(item))
            negative_indices = torch.tensor(negative_indices).cuda()
        else:
            data = torch.load(f'data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_augmented_data.pkl')
            negative_fea_tensor = data.x.to(device)
            merged_edges_ = data.edge_index.to(device)
            negative_indices=[]
            with open(f'data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_augmented_negative_indices.txt', 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    negative_indices.append(int(x))
            negative_indices = torch.tensor(negative_indices).cuda()
#
    print('input_fea_tensor', negative_fea_tensor.shape)
    print('negative_indices',len(negative_indices))
    print('merged_edges', merged_edges_.shape)
    feature_dim = negative_fea_tensor.shape[1]
    if args.model_name == 'FAGCN':
        a = (np.array(merged_edges_.cpu())[0], np.array(merged_edges_.cpu())[1])
        g = DGLGraph(a)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        g = g.to(device)

    if args.model_name == 'H2GCN':
        a = (np.array(merged_edges_.cpu())[0], np.array(merged_edges_.cpu())[1])
        g = DGLGraph(a)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        # merged_edges_modified = g.edges()
        h2gcn_adj = eidx_to_sp(negative_fea_tensor.size(0), torch.stack(g.edges(), 0)).cuda()
    if args.model_name == 'GCN':
        a = torch_geometric.utils.to_scipy_sparse_matrix(merged_edges_)
        ori_adj = process.normalize_adj(a + sp.eye(a.shape[0]))
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(ori_adj)
else:
    feature_dim = features.shape[1]
    negative_indices = None
    g = None

#

#
#
LP = False
model = PrePrompt(ft_size, args.hid_units,args.num_layers,args.drop_percent,
                  model_name=args.model_name, reduction=args.reduction,hop_level=args.hop_level, gp=args.gp,weight=args.weight,
                  g=g if args.model_name =='FAGCN'  else None,eps=args.eps,
                  aug_type = args.aug_type, sample_num = args.sample_num,inputs_indices=negative_indices)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if torch.cuda.is_available():
    print('Using CUDA')
    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
'''
------------------------------------------------------------
Pretraining
------------------------------------------------------------
'''

if not os.path.exists(args.pretrained_model):
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        if args.model_name =='FAGCN':
            loss= model(feature=negative_fea_tensor,temperature=10) # feature, shuf_fts,aug_features1,aug_features2,sp_adj,sp_aug_adj1,sp_aug_adj2,sparse,aug_type):
        elif args.model_name =='H2GCN':
            loss = model(feature=negative_fea_tensor,adj=h2gcn_adj, temperature=10)
        elif args.model_name =='DGI':
            loss = model(feature=negative_fea_tensor, temperature=10)
        else:
            loss = model(feature=negative_fea_tensor,adj=sp_adj, temperature=10)
        print('Loss:[{:.4f}]'.format(loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.pretrained_model)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
    print('Loading {}th epoch {}'.format(best_t, loss.item()))

try:
    model.load_state_dict(torch.load(args.pretrained_model))
except:
    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
model.eval()
if args.model_name == 'FAGCN':
    embeds, _ = model.embed(feature=features_origin, g=g_origin.to(features_origin.device))
if args.model_name == 'H2GCN':
    embeds, _ = model.embed(adj=sp_adj,feature=features_origin )
if args.model_name == 'GCN':
    embeds, _ = model.embed( feature=features_origin,adj=sp_adj)
if embeds.dim()==2:
    embeds=embeds.unsqueeze(0)

# preval_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
test_adj = adj.todense().A
# test_adj = test_adj[:,idx_test][idx_test,:].A
testneighbors = [[]for m in range(len(idx_test)) ]
testneighbors_2hop = [[]for m in range(len(idx_test))  ]
for step,x in enumerate(idx_test):
    tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_test[step],args.k)
    testneighbors[step] = tempneighbors
    testneighbors_2hop[step] = tempneighbors_2hop


#subgraph sampling
import torch_geometric
import dgl
graph_list = []

tot = torch.zeros(1)
tot = tot.cuda()
accs = []

print('-' * 100)

best = 1e9
best_t = 0
'''
------------------------------------------------------------
Downstream prompting
------------------------------------------------------------
'''
xent = nn.CrossEntropyLoss()
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum",shotnum)
    for i in range(100):
        idx_train = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/idx.pt").type(torch.long).cuda()
        train_lbls = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(
            torch.long).squeeze().cuda()
        print(f'NO.{i} {idx_train}')
        # print(len(idx_train))
        neighbors = [[]for m in range(len(idx_train)) ]
        neighbors_2hop = [[]for m in range(len(idx_train))  ]
        # print(neighbors)
        # train_adj = adj.todense().A
        # train_adj = train_adj[:,train_range][train_range,:].A
        for step,x in enumerate(idx_train):
            tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_train[step].item(),args.k)
            neighbors[step] = tempneighbors
            neighbors_2hop[step] = tempneighbors_2hop

        pretrain_embs = embeds[0, idx_train]


        embeds_ = embeds.squeeze(0)

        #downstream prompting
        log = downprompt(neighbors, neighbors_2hop, nb_classes, embeds, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction,multi_prompt=args.multi_prompt,
                         concat_dense=None,bottleneck_size=args.bottleneck_size,
                         meta_in=args.meta_in, dropout=args.dropout, out_size = args.out_size, activation = args.activation,
                         hidden_size=args.hid_units,prompt=args.prompt,use_metanet=args.use_metanet)

        opt = torch.optim.Adam(log.parameters(), lr=args.down_lr)
        log.cuda()
        best = 1e9
        pat_steps = 0
        cnt_wait = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(200):
            log.train()
            opt.zero_grad()

            logits = log.forward(train=1,idx=idx_train).float().cuda()
            loss = xent(logits, train_lbls)
            if not loss.requires_grad:
                loss.requires_grad = True

            if loss < best:
                best = loss
                # best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), args.pretrained_model)
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward(retain_graph=True)
            opt.step()

        logits = log.forward(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,idx=idx_test)

        preds = torch.argmax(logits, dim=1)

        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
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
