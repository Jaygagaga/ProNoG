import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import scipy.sparse as sp
from tqdm.contrib import tzip
import random
from preprompt_new import PrePrompt
# from models.FAGCN.model import FAGCN
# from models.H2GCN.model import H2GCN
# import dgl
# from dgl import DGLGraph
# import preprompt_new
# # from preprompt_metanet import PrePrompt
# # import preprompt_metanet
from utils import process
# from models.FAGCN.utils import accuracy, preprocess_data
# import pdb


import argparse
from downprompt_metanet3 import downprompt
import csv
import torch_geometric
from torch_geometric.datasets import WebKB
import tqdm
# from utils.heterophilic import WebKB, WikipediaNetwork, Actor

# from torch_geometric.loader import DataLoader
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="wisconsin", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.05, help='drop percent')
parser.add_argument('--drop_edge', type=float, default=0.2, help='drop percent')
parser.add_argument('--dropout', type=float, default=0, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--use_origin_feature',default=True,   action="store_true", help='use original feature or processed features')


parser.add_argument('--pretrained_model', type=str, default='../modelset/FAGCN_GraphCL_wisconsin_edge_sample70_.pkl',
                    help='save ckpt name')
parser.add_argument('--num_layers', type=int, default=2, help='num of layers')
parser.add_argument('--model_name', type=str, default='FAGCN', help='model')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=10, help='number of neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='pretraining learning rate')
parser.add_argument('--down_lr', type=float, default=0.0001, help='downstream learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=50, help='patience')

# downstream hyperperemeter
parser.add_argument('--prompt', type=int, default=0, help='0:noprompt, 1: use prompt')
parser.add_argument('--multi_prompt', type=int, default=0, help='1:multi prompt or 0:single prompt')
parser.add_argument('--gp', type=int, default=1, help='0: no subgraph, 1: subgraph')
parser.add_argument('--use_metanet', type=int, default=1, help='0: no metanet, 1: use metanet')
parser.add_argument('--no_prompt', type=int, default=0, help='1: no prompt, 0: have prompt')
parser.add_argument('--weight', type=int, default=1, help='0: no weights, 1: weights')
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--down_weight', type=int, default=0, help='0: no downstream weights, 1: weights')
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--sample_num', type=int, default=70, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=256, help='hidden size')
parser.add_argument('--bottleneck_size', type=int, default=64, help='bottleneck size')
parser.add_argument('--activation', type=int, default=0, help='downstream activation')
parser.add_argument('--meta_in', type=int, default=256, help='metanet input size')
parser.add_argument('--out_size', type=int, default=256, help='metanet output size')
parser.add_argument('--filename', type=str, default='no_prompt_with_metanet_origin_feature', help='save evaluation results')




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

batch_size = 1
nb_epochs = 5000
patience = 50
lr = 0.00001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
useMLP = False
import torch_geometric.transforms as T
nonlinearity = 'prelu'  # special name to separate parameters
import utils.aug as aug
from utils.heterophilic import  WikipediaNetwork, Actor
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
dataset = TUDataset(root='data', name='ENZYMES',use_node_attr=True)
# loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
if args.dataset in ['pubmed', 'citeseer', 'cora']:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    labels = torch.FloatTensor(labels[np.newaxis])
    test_lbls = torch.argmax(labels[0, idx_test], dim=1).cuda()
    val_lbls = torch.argmax(labels[0, idx_val], dim=1).cuda()
    # features, _ = process.preprocess_features(features)
    origin_adj = adj.todense()
    adj = sp.csr_matrix(adj)
    features = process.sparse_mx_to_torch_sparse_tensor(features).to_dense()
    idx_test= torch.tensor(idx_test).cuda()
    idx_val = torch.tensor(idx_val).cuda()
if args.dataset in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
    datasets = HeterophilousGraphDataset(root='data', name=args.dataset)
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    origin_adj = adj.todense().A
# if args.dataset =='cornell':
if args.dataset in ['cornell', 'texas', 'wisconsin']:
    datasets = WebKB(root='data', name=args.dataset)
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    # idx_test = range(features.shape[0]-100,features.shape[0])
    # origin_adj = adj.todense().A
    origin_adj = adj.todense().A
    if args.dataset == 'cornell':
        origin_adj = adj.todense().A.T
if args.dataset in ['chameleon', 'squirrel']:
    datasets = WikipediaNetwork(root='data', name=args.dataset)
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
if args.dataset in ['film']:
    datasets = Actor(root='data')
    features, adj, labels, sizes, masks = process.process_webkb(datasets.data, datasets.data.x.shape[0])
    origin_adj = adj.todense().A

# features= process.preprocess_features(features)
# features = torch.FloatTensor(features[np.newaxis])

idx_test = torch.load(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/test_idx.pt").type(torch.long).cuda()
test_lbls = torch.load(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/test_labels.pt").type(torch.long).cuda()
# if args.dataset == 'cornell':
#     origin_adj = adj.todense().A.T
# else:
#     origin_adj = adj.todense().A
# if args.dataset == 'cora':
#     origin_adj = adj.todense()
from torch_geometric.utils.convert import from_scipy_sparse_matrix
edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
# edge_index = from_scipy_sparse_matrix(adj)[0].cuda()

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)




# negetive_sample = preprompt_new.prompt_pretrain_sample(adj,args.sample_num)
# negetive_sample =preprompt_metanet.prompt_pretrain_sample(adj,70)
nb_nodes = features.shape[0]  # node number
ft_size = features.shape[-1]  # node features dim
nb_classes = labels.shape[-1]  # classes = 6
device="cuda"
import dgl
from models.FAGCN.utils import normalize_features
if args.model_name =='FAGCN':
    from models.FAGCN.utils import preprocess_data
    if args.dataset in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions","pubmed", "citeseer", "cora"]:
        features_origin = normalize_features(features)
        features_origin = torch.FloatTensor(features_origin).cuda()
        from torch_geometric.utils.convert import from_scipy_sparse_matrix

        
        g_origin = dgl.graph((np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1]))
        g_origin = dgl.to_simple(g_origin)
        g_origin = dgl.remove_self_loop(g_origin)
        g_origin = dgl.to_bidirected(g_origin)
        g_origin = g_origin.to(features_origin.device)
        # deg = g_origin.in_degrees().cuda().float().clamp(min=1)
        # norm = torch.pow(deg, -0.5)
        # g_origin.ndata['d'] = norm
    else:
        g_origin, nclass, features_origin, labels_origin = preprocess_data(args.dataset, 1)
        # features_origin = features.cuda()
        labels_origin = labels_origin.cuda()
    

    deg = g_origin.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g_origin = g_origin.to(labels_origin.device)
    g_origin.ndata['d'] = norm

features = features.cuda()
if args.model_name =='H2GCN':
    from models.H2GCN.utils import load_dataset
    features_origin, _, _, _, adj_origin= load_dataset(args.dataset,device)
if args.model_name =='GCN':
    features_origin = features.squeeze(0)
    adj_origin = sp_adj
if args.use_origin_feature ==True:
    features_origin = features.cuda()
else:
    features_origin = features_origin.cuda()
# if args.model_name =='FAGCN' or args.model_name =='GraphCL':
#     from models.FAGCN.utils import preprocess_data
#     g, nclass, fagcn_features, f2gcn_labels = preprocess_data(args.dataset, 1)
#     fagcn_features = fagcn_features.cuda()
#     f2gcn_labels = f2gcn_labels.cuda()
#
#     g = g.to(fagcn_features.device)
#     deg = g.in_degrees().cuda().float().clamp(min=1)
#     norm = torch.pow(deg, -0.5)
#     g.ndata['d'] = norm
# else:
#     fagcn_features = None
#     g=None
# features = torch.FloatTensor(features[np.newaxis])
# fagcn_features = torch.FloatTensor(fagcn_features[np.newaxis])
"""
------------------------------------------------------------
2hop neighbors and subgraphs
------------------------------------------------------------
"""
if not os.path.exists(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/neighbors_2hops.csv"):
    neighbors = []
    neighbors_2hops = []
    neighborslist = [[] for x in range(adj.shape[0])]
    neighbors_2hoplist = [[] for x in range(adj.shape[0])]
    neighborsindex = [[] for x in range(adj.shape[0])]
    neighbors_2hopindex = [[] for x in range(adj.shape[0])]
    # neighboradj = adj.todense().A
    for x in tqdm.trange(adj.shape[0]):
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

# valneighbors = [[] for m in range(len(idx_val))]
# valneighbors_2hop = [[] for m in range(len(idx_val))]
# for step, x in enumerate(idx_val):
#     tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,
#                                                                     idx_val[step].item(), args.k)
#     valneighbors[step] = tempneighbors
#     valneighbors_2hop[step] = tempneighbors_2hop

'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
import ast
import utils.aug1 as aug
def augmentation(subfeatures,adj,aug_type,drop_edge=0.2):
    if aug_type == 'edge':
        aug_features1 = subfeatures

        aug_adj1 = aug.aug_random_edge(adj, drop_edge=drop_edge)  # random drop edges

    elif aug_type == 'mask':

        aug_features1 = aug.aug_random_mask(subfeatures, drop_edge=drop_edge)

        aug_adj1 = adj

    else:
        assert False
    return aug_features1,aug_adj1
# if os.path.exists(args.pretrained_model):
#     subgraphs = None
#     subgraph_logits_ids=None
#     negative_samples=None

print("Begin Aug:[{}]".format(args.aug_type))
subgraphs = []
subfeatures = []
subadjs = []
for node in tqdm.trange(nb_nodes):
    subset_nodes = [node] + neighbors[node] + neighbors_2hops[node]
    new_edge_index, _ = torch_geometric.utils.subgraph(edge_index=edge_index.cpu(), subset=subset_nodes,
                                                       relabel_nodes=False)
    new_adj = torch_geometric.utils.to_dense_adj(new_edge_index).squeeze(0)
    subadjs.append(sp.csr_matrix(new_adj))
    # a = (torch.tensor(new_edge_index.tolist()[0]), new_edge_index.tensor(edge_index.tolist()[1]))
    # g = DGLGraph(a)
    # g = dgl.to_simple(g)
    # g = dgl.to_bidirected(g)
    # g = dgl.remove_self_loop(g)
    subgraphs.append(subset_nodes)
    subfeatures.append(features[subset_nodes])
negative_samples = []
negative_sample_idx = []
# negative_subsets = []
# torch.save(subgraphs,
#            f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subgraphs.pkl')
# torch.save(subfeatures,
#            f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subfeatures.pkl')
# torch.save(subadjs,
#            f'./data/{args.dataset}/{args.aug_type}_{args.sample_num}_{str(args.drop_edge)[-1]}_subadjs.pkl')

for num, (subset,subadj) in enumerate(tzip(subfeatures, subadjs)):
    negative_sample_=[]
    negative_sample_id=[]
    rest = [id for id in range(nb_nodes) if id != num]
    random.shuffle(rest)
    if args.aug_type =='edge':
        aug_ = augmentation(subset, subadj, aug_type=args.aug_type,drop_edge=args.drop_edge)[-1]

        negative_sample_.append(aug_)


        for i in rest:
            if len(negative_sample_) ==args.sample_num+1:
                break
            if subadjs[i].count_nonzero() != 0:
                negative_sample_ += [subadjs[i]]
    # if args.aug_type == 'mask':
    #     aug_ = augmentation(subset, subadj, aug_type=args.aug_type)[0]
    #     negative_sample_.append(aug_)
    #     negative_sample_ += [subset[i] for i in rest[:args.sample_num]]

    negative_samples.append(negative_sample_)

# print(negative_samples[3][0].nonzero())
# print(subadjs[3].nonzero())
subgraph_logits_ids = [[]*(args.sample_num+1)  for x in range(adj.shape[0])]
# subgraph_logits_ids = subgraphs
for num, (sample, graph) in enumerate(tzip(negative_samples, subgraphs)):
    # subgraphs: subsets of 2hop graph nodes, sample: lists of lists of adj(一个node有长度为args.sample_num+1的adj list,第0个是增强positive，剩下的70个是其他subgrah的adj)

    for i in range(len(sample)):
        # if args.reduction == 'mean':
            if len(graph) == 1:  # the node does not have neighbors, self embedding as sugraph embedding
                # self_logits_ids[num] += [num]

                if i == 0:
                    subgraph_logits_ids[num] += [num] # positive sample = self
                else:
                    subgraph_logits_ids[num] +=[np.unique(np.array(list(sample[i].nonzero())).flatten())]

            else:  # the node does have neighbors
                # self_logits_ids[num] =graph
                subgraph_logits_ids[num] += [np.unique(np.array(list(sample[i].nonzero())).flatten())] # 用最开始的logits_, 每个negative subgraph samples整合的subgraph embedding
negative_edge_index=[]
for n in negative_samples:
    for i in n:
        negative_edge_index.append(from_scipy_sparse_matrix(i)[0])
if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)

# labels = torch.FloatTensor(labels[np.newaxis])
# print("labels",labels)
print("adj",sp_adj.shape)
print("feature",features.shape)

"""
------------------------------------------------------------
Pretraining
------------------------------------------------------------
"""
LP = True
model = PrePrompt(ft_size, args.hid_units,args.num_layers,0.05,
                              model_name=args.model_name, reduction=args.reduction,hop_level=args.hop_level, gp=args.gp,weight=args.weight,
                              concat_dense=args.concat_dense,pretrain_hop = args.pretrain_hop, g=g_origin if args.model_name =='FAGCN' else None,eps=args.eps,
                            subgraphs = subgraphs, aug_type = args.aug_type,subgraph_logits_ids=subgraph_logits_ids,sample_num = args.sample_num,sample=negative_samples)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if torch.cuda.is_available():
    print('Using CUDA')
    model = model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    # labels = labels.cuda()
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
model=model.cuda()
if args.model_name == 'GCN':
    features_origin = features_origin.unsqueeze(0)
if os.path.exists(args.pretrained_model):
    model.load_state_dict(torch.load(args.pretrained_model), strict=False)
    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best = checkpoint['loss']
else:
    epoch = -1

for epoch_ in range(epoch+1,nb_epochs):
    model.train()
    optimiser.zero_grad()

    if args.model_name =='FAGCN':
        loss= model(seq=features_origin,temperature=10)
    if args.model_name =='H2GCN' or args.model_name =='GCN'  :
        loss= model(seq=features_origin.unsqueeze(0).cuda() if features_origin.dim()==2 else features_origin, adj = adj_origin.cuda(),temperature=10) # feature, shuf_fts,aug_features1,aug_features2,sp_adj,sp_aug_adj1,sp_aug_adj2,sparse,aug_type):
    print('Loss:[{:.4f}]'.format(loss.item()))
    if loss < best:
        best = loss
        best_t = epoch_
        cnt_wait = 0
        # torch.save(model.state_dict(), args.pretrained_model)
        torch.save({
            'epoch': epoch_,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss': loss.item(),
        }, args.pretrained_model)
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break
    loss.backward()
    optimiser.step()
# # model.load_state_dict(torch.load(args.pretrained_model),strict=False)
checkpoint = torch.load(args.pretrained_model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
if args.model_name =='FAGCN':
    embeds, _ = model.embed(seq=features_origin)
if args.model_name =='H2GCN' or args.model_name =='GCN':
    embeds, _ = model.embed(seq=features_origin, adj = adj_origin)
if embeds.dim()==2:
    embeds=embeds.unsqueeze(0)
# preval_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]
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




"""
------------------------------------------------------------
Downstream prompting
------------------------------------------------------------
"""
print('-' * 100)

for shotnum in args.shotnum:

    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum",shotnum)
    for i in range(100):
        idx_train = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/idx.pt").type(torch.long).cuda()
        print(f'NO.{i} {idx_train}',)
        # print(len(idx_train))
        neighbors = [[]for m in range(len(idx_train)) ]
        neighbors_2hop = [[]for m in range(len(idx_train))  ]
        # print(neighbors)
        train_adj = adj.todense().A
        # train_adj = train_adj[:,train_range][train_range,:].A
        for step,x in enumerate(idx_train):
            tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_train[step].item(),args.k)
            neighbors[step] = tempneighbors
            neighbors_2hop[step] = tempneighbors_2hop

        pretrain_embs = embeds[0, idx_train]
        train_lbls = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(torch.long).squeeze().cuda()

        embeds_ = embeds.squeeze(0)
        cnt_wait = 0
        best = 1e9
        best_t = 0
        # # 下游supervised
        # if args.model_name == 'FAGCN':
        #
        #     downstream_model = FAGCN(g_origin, features_origin.size()[1], args.hid_units, nb_classes, args.dropout, args.eps, args.num_layers,
        #          ).cuda()
        #     dict = downstream_model.state_dict()
        #     pretrained_dict = {k: v for k, v in model.model.state_dict().items() if k in dict and 't2' not in k}
        #     dict.update(pretrained_dict)
        #     downstream_model.load_state_dict(dict)
        #     for n, m in downstream_model.named_parameters():
        #         if 'gate' in n or 't1' in n:
        #             m.requires_grad = False  # 下游supervised 继续训练/不继续训练
        # if args.model_name == 'H2GCN':
        #     downstream_model = H2GCN(features_origin.size()[1],hidden_dim=args.hid_units,class_dim=args.hid_units,use_relu=False).cuda()
        #     dict = downstream_model.state_dict()
        #     pretrained_dict = {k: v for k, v in model.model.state_dict().items() if k in dict}
        #     dict.update(pretrained_dict)
        #     downstream_model.load_state_dict(dict)
        #     for n, m in downstream_model.named_parameters():
        #         if 'w_embed' in n:
        #             m.requires_grad = False  # 下游supervised 继续训练/不继续训练
        #
        #
        # # model.load_state_dict(torch.load(args.pretrained_model))
        # opt = torch.optim.Adam(downstream_model.parameters(), lr=0.0001)
        # min_loss = 100
        # best_t = 0
        # for _ in range(200):
        #     downstream_model.train()
        #     opt.zero_grad()
        #     if args.model_name == 'H2GCN':
        #         logp = downstream_model(seq=features_origin,adj=adj_origin)
        #     if args.model_name == 'FAGCN':
        #         logp = downstream_model(features_origin)
        #     # print(logp[idx_train])
        #     cla_loss = F.nll_loss(logp[idx_train], train_lbls)
        #     loss = cla_loss
        #     if loss < min_loss:
        #         min_loss = loss
        #         best_t = _
        #         # max_acc = val_acc
        #         counter = 0
        #     else:
        #         counter += 1
        #
        #     if counter >= args.patience:
        #         print('early stop')
        #         break
        #     # accs.append(acc * 100)
        #     # print('acc:[{:.4f}]'.format(acc))
        #     # tot += acc
        #     loss.backward()
        #     opt.step()
        #
        # # model.eval()
        # acc = accuracy(logp[idx_test], test_lbls)

        # #下游prompt
        # fagcn_features = None
        # g = None
        #
        #
        log = downprompt(neighbors, neighbors_2hop, nb_classes, embeds, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction,multi_prompt=args.multi_prompt,
                         concat_dense=None,bottleneck_size=args.bottleneck_size,pretrain_weights=model.tokens,
                         meta_in=args.meta_in, dropout=args.dropout, out_size = args.out_size, activation = args.activation,
                          hidden_size=args.hid_units,use_metanet = args.use_metanet, prompt=args.prompt)

        opt = torch.optim.Adam(log.parameters(), lr=args.down_lr)
        log.cuda()
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(200):
            log.train()
            opt.zero_grad()

            logits = log(train=1,idx=idx_train).float().cuda()
            loss = xent(logits, train_lbls)

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
            # log.eval()
            # with torch.no_grad():
            #     logits_val = log.forward(train=0, neighbors=valneighbors, neighbors_2hop=valneighbors_2hop,
            #                              idx=idx_val)
            #     loss_val = xent(logits_val, val_lbls)
            #     val_preds = torch.argmax(logits_val, dim=1)  # 获取预测结果
            #     val_acc = torch.sum(val_preds == val_lbls).float() / val_lbls.shape[0]  # 计算准确率
            #
            # if val_acc > best:
            #     best = val_acc
            #     # best_t = epoch
            #     cnt_wait = 0
            #     # torch.save(model.state_dict(), args.save_name)
            # else:
            #     cnt_wait += 1
            # if cnt_wait == patience:
            #     print('Early stopping!')
            #     break
        # prompt_feature = feature_prompt(features)
        # embeds1, _ = model.embed(prompt_feature, sp_adj if sparse else adj, sparse, None,LP)
        # test_embs1 = embeds1[0, idx_test]
        logits = log(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,idx=idx_test)
        # logits = log(train=0,idx=idx_test_,origin_adj=origin_adj_)
        # print("logits",logits)
        # print(log.a)
        preds = torch.argmax(logits, dim=1)
        # print('----predcited labels:',preds)
        # print('----test labels:', test_lbls)
        # print('----degree:', [deg[i].item() for i in idx_test])
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
    df = pd.DataFrame(list(zip([args.model_name],[args.pretrained_model],[args.num_layers], [args.weight],[args.lr],[args.down_lr],[args.weight_decay],[args.reduction],[args.hop_level],[args.gp],[args.drop_percent],[args.dropout],[args.drop_edge], [args.aug_type],[args.prompt], [args.multi_prompt],[args.k],
                               [args.sample_num],[args.hid_units],[args.bottleneck_size],[args.meta_in],[accs.mean().item()],[accs.std().item()],[shotnum])),
    columns= ['model_name' ,'pretrained_model','num_layers', 'weight', 'lr','down_lr','weight_decay','reduction', 'hop level','gp','drop_percent','dropout','drop_edge','aug_type','prompt','multi_prompt','k', 'sample_num', 'hid_units', 'bottleneck_size','meta_in', 'accuracy', 'mean', 'shotnum'])
    if os.path.exists(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv"):
        df.to_csv(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv",mode='a',header=False)
    else:
        df.to_csv(f"data/{args.dataset}/{args.model_name}_{args.filename}.csv")
    # row = [shotnum,args.lr,args.hid_units,accs.mean().item(),accs.std().item()]
    # filename = f"data/{args.dataset}/{args.model_name}_layer{args.num_layers}_weight{args.weight}_{args.reduction}_hop{args.hop_level}_gp{args.gp}_prompt{args.prompt}_multi{args.multi_prompt}_.csv"
    # out = open(filename, "a", newline="")
    # csv_writer = csv.writer(out, dialect="excel")
    # csv_writer.writerow(row)
