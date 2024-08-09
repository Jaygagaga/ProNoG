import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from graphcl.dgi import DGI
from graphcl.logreg import LogReg
import graphcl.aug as aug
from utils import process
import pdb
import graphcl.aug
import os
import argparse
import pandas as pd
from downprompt_metanet3 import downprompt

parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset',          type=str,           default="BZR",                help='data')
parser.add_argument('--aug_type',         type=str,           default="edge",                help='aug type: mask or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.2,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
# parser.add_argument('--gpu',              type=int,           default=1,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='modelset/graphcl/BZR_graphcl4.pkl',                help='save ckpt name')


parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--down_lr', type=float, default=0.0001, help='lr for downstream')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=50, help='patience')
parser.add_argument('--nb_epochs', type=int, default=2000, help='number of epochs')
parser.add_argument('--model_name', type=str, default='GCN', help='choose backbone: FAGCN/GCN')
# downstream hyperperemeter
parser.add_argument('--use_origin_feature', type=int, default=1, help='')
parser.add_argument('--prompt', type=int, default=0, help='0:no prompt,1:use prompt')
parser.add_argument('--multi_prompt', type=int, default=0, help='1:multi prompt or 0:single prompt')
parser.add_argument('--use_metanet', type=int, default=1, help='use metanet layer or not')
parser.add_argument('--meta_in', type=int, default=256, help='number of neighbors')
parser.add_argument('--bottleneck_size', type=int, default=64, help='number of neighbors')
parser.add_argument('--out_size', type=int, default=256, help='number of neighbors')
parser.add_argument('--gp', type=int, default=1, help='0: no subgraph, 1: subgraph')
parser.add_argument('--weight', type=int, default=1, help='0: no weights, 1: weights')
parser.add_argument('--reduction', type=str, default='mean', help='reduction')
parser.add_argument('--concat_dense', type=int, default=0, help='1: apply concat then linear')
parser.add_argument('--down_weight', type=int, default=0, help='0: no downstream weights, 1: weights')
parser.add_argument('--shotnum', type=list, default=[1], help='shot num')
parser.add_argument('--sample_num', type=int, default=10, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=256, help='number of neighbors')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='no_prompt_with_metanet_origin_feature_graphcl_graph2', help='number of neighbors')
args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params


batch_size = 1
nb_epochs = 10000
patience = 50
lr = 0.00001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
import dgl
from torch_geometric.datasets import WebKB
from utils.heterophilic import  WikipediaNetwork, Actor
import torch_geometric
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
def edgeindex2graph(edge_index):
    g = dgl.graph((np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1]))
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm
    g = g.to(edge_index.device)
    return g


import ast
if args.dataset in ['BZR','COX2']:

    with open(f"/home/xingtong/WebKB_node_origin/data/{args.dataset}/{args.dataset}.edges") as f:
        edges_index = [ast.literal_eval(i) for i in  f.readlines()]
    with open(f"/home/xingtong/WebKB_node_origin/data/{args.dataset}/{args.dataset}.graph_idx") as f:
        graph_labels =[ast.literal_eval(i) for i in  f.readlines()]
    with open(f"/home/xingtong/WebKB_node_origin/data/{args.dataset}/{args.dataset}.node_attrs") as f:
        features= np.array([ast.literal_eval(i) for i in f.readlines()])
    with open(f"/home/xingtong/WebKB_node_origin/data/{args.dataset}/{args.dataset}.node_labels") as f:
        labels= torch.tensor([ast.literal_eval(i) for i in f.readlines()]).cuda()

    # features = process.preprocess_features(features)
    a = torch_geometric.utils.to_scipy_sparse_matrix(torch.tensor(edges_index).T)
    adj = sp.csr_matrix(a)
    origin_adj = np.array(adj.todense())
    if args.use_origin_feature == 0:
        features = process.preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis])
    nb_nodes = features.size(1)
    g = edgeindex2graph(torch.tensor(edges_index).cuda())
    edge_index = torch.tensor(origin_adj).nonzero().t().contiguous()
    count = 0
    for i, j in zip(edge_index[0],edge_index[1]):
        if labels[i].item() == labels[j].item():
            count += 1
    homo_ratio =count/edge_index.shape[1]
    print(homo_ratio)


if args.dataset in ['ENZYMES','PROTEINS']:
    data = torch.load(f"data/{args.dataset}/data.pkl")
    features = data.features
    adj = data.adj
    origin_adj = np.array(adj.todense())
    labels = data.labels
    edge_index = data.edge_index
    # features = process.preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis]).cuda()
if args.dataset in ['ENZYMES','PROTEINS','BZR','COX2']:
    path = f"data/fewshot_{args.dataset}_graph"
else:
    path = f"data/fewshot_{args.dataset}"
test_features = torch.load(f"{path}/testset/feature.pt")
if args.use_origin_feature ==0:
    test_features=process.preprocess_features(test_features)
test_features = torch.FloatTensor(test_features[np.newaxis]).cuda()
test_adj =torch.load(f"{path}/testset/adj.pt")
test_edge_index = test_adj.nonzero().t().contiguous().cuda()
test_adj_ = test_adj.numpy()
adj_csr = sp.csr_matrix(test_adj_)
test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()
test_lbls = torch.load(f"{path}/testset/labels.pt")
nb_classes = len(torch.unique(test_lbls))
graph_len = torch.load(f"{path}/testset/graph_len.pt")
test_g = edgeindex2graph(test_edge_index)
import tqdm
testneighbors = [[] for m in range(len(test_adj_))]
testneighbors_2hop = [[] for m in range(len(test_adj_))]
for step, x in enumerate(range(test_adj_.shape[0])):
    tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(test_adj_, x)
    testneighbors[step] = tempneighbors
    testneighbors_2hop[step] = tempneighbors_2hop




# origin_datasets = torch.load(f'/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}_graph/graph_data1.pkl')
# graph_idx =[i.graph_nodes[0] for i in origin_datasets]
# graph_label = [i.y for i in origin_datasets]
# idx_test = torch.load(f"data/fewshot_{args.dataset}_graph/test_idx.pt").type(torch.long).cuda()
# test_lbls = torch.load(f"data/fewshot_{args.dataset}_graph/test_labels.pt").type(torch.long)
# test_datasets = [origin_datasets[t] for t in idx_test]
# graph_idx_test = [i.graph_nodes[0] for i in test_datasets]
# graph_label_test = [i.y for i in test_datasets]
# test_loader = DataLoader(test_datasets, batch_size=args.batch_size, follow_batch=['x'] * args.batch_size,shuffle=False)
ft_size = test_features.size(-1)

'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))
if args.aug_type == 'edge':
    if not os.path.exists(f"data/{args.dataset}/aug_adj2_{str(args.drop_percent)}.pkl"):

        aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
        aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
        torch.save(aug_adj1, f"data/{args.dataset}/aug_adj1_{str(args.drop_percent)}.pkl")
        torch.save(aug_adj2, f"data/{args.dataset}/aug_adj2_{str(args.drop_percent)}.pkl")
    else:
        aug_adj1 = torch.load(f"data/{args.dataset}/aug_adj1_{str(args.drop_percent)}.pkl")
        aug_adj2 = torch.load(f"data/{args.dataset}/aug_adj2_{str(args.drop_percent)}.pkl")
    aug_features1 = features
    aug_features2 = features
    if args.model_name == 'FAGCN':

        edge_index1 = torch.tensor(aug_adj1.todense()).nonzero().t().contiguous().cuda()
        g1= edgeindex2graph(edge_index1)
        edge_index2 = torch.tensor(aug_adj2.todense()).nonzero().t().contiguous().cuda()
        g2 = edgeindex2graph(edge_index2)

elif args.aug_type == 'node':

    aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'subgraph':

    aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'mask':
    if not os.path.exists(f"data/{args.dataset}/aug_features1_{str(args.drop_percent)}.pkl"):

        aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
        aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)
        torch.save(aug_features1, f"data/{args.dataset}/aug_features1_{str(args.drop_percent)}.pkl")
        torch.save(aug_features2, f"data/{args.dataset}/aug_features2_{str(args.drop_percent)}.pkl")
    else:
        aug_features1 = torch.load(f"data/{args.dataset}/aug_features1_{str(args.drop_percent)}.pkl")
        aug_features2 = torch.load(f"data/{args.dataset}/aug_features2_{str(args.drop_percent)}.pkl")


    aug_adj1 = adj
    aug_adj2 = adj

else:
    assert False


#
#
# '''
# ------------------------------------------------------------
# '''
#
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
    aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
    aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()




xent = nn.CrossEntropyLoss()
model = DGI(ft_size, hid_units, 'prelu',model_name = args.model_name)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    aug_features1 = aug_features1.cuda()
    aug_features2 = aug_features2.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()
    else:
        adj = adj.cuda()
        aug_adj1 = aug_adj1.cuda()
        aug_adj2 = aug_adj2.cuda()

b_xent = nn.BCEWithLogitsLoss()

cnt_wait = 0
best = 1e9
best_t = 0
if not os.path.exists(args.save_name):
    for epoch in range(nb_epochs):

        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        if args.model_name=='GCN':
            logits = model(features, shuf_fts, aug_features1, aug_features2,
                           sp_adj if sparse else adj,
                           sp_aug_adj1 if sparse else aug_adj1,
                           sp_aug_adj2 if sparse else aug_adj2,
                           sparse, None, None, None, aug_type=aug_type)
        if args.model_name=='FAGCN':
            logits = model(features.squeeze(0), shuf_fts.squeeze(0), aug_features1, aug_features2,
                           sp_adj if sparse else adj,
                           sp_aug_adj1 if sparse else aug_adj1,
                           sp_aug_adj2 if sparse else aug_adj2,
                           sparse, None, None, None, aug_type=aug_type, g= g,g1=g1,g2=g2)


        loss = b_xent(logits, lbl)
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
model.cuda()
if args.model_name=='GCN':
    test_embeds, _ = model.embed(test_features, sp_adj_test, sparse,None)
if args.model_name=='FAGCN':
    test_embeds, _ = model.embed(test_features.squeeze(0) if test_features.dim()==3 else test_features, test_g)

if test_embeds.dim() == 3:
    test_embeds = test_embeds.squeeze(0)

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
for shotnum in args.shotnum:
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print("shotnum",shotnum)
    for i in range(100):
        print('No.{} task'.format(i))
        # if args.dataset == 'ENZYMES':
        #     idx_train = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodeidx.pt").type(torch.long).cuda()
        #     train_lbls = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/nodelabels.pt").type(torch.long).squeeze().cuda()
        # if args.dataset in ['BZR','COX2'] :
        adj_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/adj.pt")
        train_edge_index = adj_train.nonzero().t().contiguous().cuda()
        train_g = edgeindex2graph(train_edge_index)
        feature_train = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/feature.pt")
        train_lbls = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(torch.long).squeeze().cuda()
        train_graph_len = torch.load(f"{path}/{shotnum}-shot_{args.dataset}/{i}/graph_len.pt")
        if args.use_origin_feature ==0:
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

        # neighbors = [[]for m in range(len(idx_train)) ]
        # neighbors_2hop = [[]for m in range(len(idx_train))  ]
        # for step,x in enumerate(idx_train):
        #     tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_train[step].item())
        #     neighbors[step] = tempneighbors
        #     neighbors_2hop[step] = tempneighbors_2hop
        if args.model_name == 'GCN':
            embeds, _ = model.embed(feature_train, sp_adj_train,sparse, None)
        if args.model_name == 'FAGCN':
            embeds, _ = model.embed(feature_train.squeeze(0) if feature_train.dim()==3 else feature_train, train_g)
        if embeds.dim()==3:
            embeds = embeds.squeeze(0)
        print("true",i,train_lbls)
        # feature_prompt=featureprompt(ft_size).cuda()
        # log = downprompt(neighbors,neighbors_2hop, hid_units, nb_classes,embeds,train_lbls,model.tokens.weight[0][0],model.tokens.weight[0][1],model.tokens.weight[0][2])

        # log = downprompt(neighbors, neighbors_2hop, hid_units, nb_classes, embeds, train_lbls,
        #                  hop_level=args.hop_level, reduction=args.reduction,multi_prompt=args.multi_prompt)
        log = downprompt(neighbors, neighbors_2hop, nb_classes, embeds, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction, multi_prompt=args.multi_prompt,
                         concat_dense=None, bottleneck_size=args.bottleneck_size,
                         meta_in=args.meta_in,  out_size=args.out_size,
                         hidden_size=hid_units, prompt=args.prompt, use_metanet=args.use_metanet)
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
            logits=log.forward_homo_graph(train=1,train_embeds= embeds,graph_len=train_graph_len).float().cuda()
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
            logits = log.forward_homo_graph(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,test_embeds=test_embeds, graph_len=graph_len)
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
        zip([args.model_name], [args.save_name],  [args.hop_level],[args.aug_type],[str(args.drop_percent)],
            [args.multi_prompt], [args.prompt], [args.use_metanet], [args.bottleneck_size],
            [accs.mean().item()], [accs.std().item()], [shotnum], [args.use_origin_feature])),
        columns=['model_name', 'pretrained_model', 'hop_level', 'aug_type','drop_percent','multi_prompt',
                 'prompt', 'metanet', 'bottleneck','accuracy', 'mean', 'shotnum','use_origin_feature'])
    if os.path.exists(f"data/{args.dataset}/{args.filename}.csv"):
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv", mode='a', header=False)
    else:
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv")

# #
# #
