import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
parser.add_argument('--dataset',          type=str,           default="cora",                help='data')
parser.add_argument('--aug_type',         type=str,           default="edge",                help='aug type: mask or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.2,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
parser.add_argument('--gpu',              type=int,           default=1,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='modelset/graphcl/cora_graphcl4.pkl',                help='save ckpt name')

parser.add_argument('--use_origin_feature', type=int, default=0, help='')
parser.add_argument('--hop_level', type=int, default=2, help='hop_level')
parser.add_argument('--k', type=int, default=20, help='number of neighbors')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--down_lr', type=float, default=0.0001, help='lr for downstream')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--patience', type=float, default=50, help='patience')
parser.add_argument('--nb_epochs', type=int, default=2000, help='number of epochs')
parser.add_argument('--model_name', type=str, default='GCN', help='choose backbone: FAGCN/GCN')
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
parser.add_argument('--sample_num', type=int, default=10, help='number of negative samples')
parser.add_argument('--hid_units', type=int, default=256, help='hidden size')
parser.add_argument('--activation', type=int, default=0, help='use activation or not')
parser.add_argument('--filename', type=str, default='no_prompt_with_metanet_origin_feature_graphcl2', help='number of neighbors')
args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params


batch_size = 1
nb_epochs = 10000
patience = 50
lr = 0.0001
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
from torch_geometric.datasets import HeterophilousGraphDataset
# if args.dataset =='cornell':
import csv
if args.dataset in ['ENZYMES', 'PROTEINS']:
    dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True, )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True)

    if not os.path.exists(f"data/{args.dataset}/data.pkl"):
        dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True,)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=True)
        for step, data in enumerate(loader):
            features, adj, labels = process.process_tu(data, 18) #1-{args.dataset}, 1-PROTEINS
            # features = dataset.data.x
            # edge_index=dataset.data.edge_index
            # labels = dataset.data.y
            # nb_class = len(torch.unique(labels))
            # coo = sp.coo_matrix((np.ones(edge_index.shape[1]),(edge_index[0, :], edge_index[1, :])), shape=(features.size(0), features.size(0)))
            # adj = coo.todense()
            adj_ = adj.todense()
            origin_adj = np.array(adj_)
            edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
        data = Data(features=features,adj =adj,labels = labels,edge_index =edge_index)
        torch.save(data, f"data/{args.dataset}/data.pkl")
    else:
        data = torch.load(f"data/{args.dataset}/data.pkl")
        features = data.features
        ft_size = features.shape[-1]
        adj = data.adj
        origin_adj =np.array(adj.todense())
        labels = data.labels
        labels_ = torch.argmax(torch.tensor(labels),dim=-1)
        edge_index = data.edge_index
        # tuple =[(i, j) for i, j in zip(edge_index[0].tolist(),edge_index[1].tolist() )]
        # count = 0
        # for num, t in enumerate(tuple):
        #     if labels_[t[0]] == labels_[t[1]]:
        #         count += 1
        # homo_ratio = count/len(tuple)
        if args.use_origin_feature == 0:
            features = process.preprocess_features(features)
        features_ = torch.FloatTensor(features[np.newaxis]).cuda()
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        sp_adj_ = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_adj_ = sp_adj_.to_dense().cuda()
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
        nb_classes = len(torch.unique(test_lbls))

if args.dataset in ['pubmed', 'citeseer', 'cora']:
    adj_node, features, labels_node, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    if args.use_origin_feature == 0:
        features = process.preprocess_features(features)
    origin_adj = adj_node.todense()
    adj_node = sp.csr_matrix(adj_node)
    adj = adj_node
    edge_index = from_scipy_sparse_matrix(adj)[0].cuda()
    features = process.sparse_mx_to_torch_sparse_tensor(features).to_dense()
    idx_test = torch.tensor(idx_test)
    labels = labels_node
    test_lbls = torch.argmax(torch.tensor(labels_node)[idx_test], dim=1)
    # g = dgl.graph((np.array(edge_index.cpu())[0], np.array(edge_index.cpu())[1]))
    idx_test = idx_test.cuda()
    test_lbls = test_lbls.cuda()
    edge_index = torch.tensor(adj.todense()).nonzero().t().contiguous().cuda()
    g = edgeindex2graph(edge_index)


features = torch.FloatTensor(features[np.newaxis])


'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))

if args.aug_type == 'edge':
    if not os.path.exists(f"data/{args.dataset}/aug_adj1.pkl"):

        aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
        aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
        torch.save(aug_adj1, f"data/{args.dataset}/aug_adj1.pkl")
        torch.save(aug_adj2, f"data/{args.dataset}/aug_adj2.pkl")
    else:
        aug_adj1 = torch.load(f"data/{args.dataset}/aug_adj1.pkl")
        aug_adj2 = torch.load(f"data/{args.dataset}/aug_adj2.pkl")
    aug_features1 = features
    aug_features2 = features
    if args.model_name == 'FAGCN':
        # edge_index = torch.tensor(adj.todense()).nonzero().t().contiguous().cuda()
        # g = edgeindex2graph(edge_index)
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
    if not os.path.exists(f"data/{args.dataset}/aug_features1.pkl"):

        aug_features1 = aug.aug_random_mask(features, drop_percent=drop_percent)
        aug_features2 = aug.aug_random_mask(features, drop_percent=drop_percent)
        torch.save(aug_features1, f"data/{args.dataset}/aug_features1.pkl")
        torch.save(aug_features2, f"data/{args.dataset}/aug_features2.pkl")
    else:
        aug_features1 = torch.load(f"data/{args.dataset}/aug_features1.pkl")
        aug_features2 = torch.load(f"data/{args.dataset}/aug_features2.pkl")

    aug_adj1 = adj
    aug_adj2 = adj

else:
    assert False

ft_size = features.size(-1)
nb_nodes = features.size(1)
nb_classes = labels.shape[-1]

'''
------------------------------------------------------------
'''

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


'''
------------------------------------------------------------
mask
------------------------------------------------------------
'''

'''
------------------------------------------------------------
'''
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
    aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])


# labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, 'prelu',model_name=args.model_name)
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

    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
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
model.load_state_dict(torch.load(args.save_name),strict=False)

model.eval()
model.cuda()
if args.model_name == 'GCN':
    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
if args.model_name == 'FAGCN':
    embeds, _ = model.embed(features.squeeze(0) if features.dim()==3 else features,g=g)
# embeds = model.logits
# CUDA_LAUNCH_BLOCKING = 1
if embeds.dim() == 3:
    embeds = embeds.squeeze(0)
test_embs=torch.index_select(embeds,0,torch.tensor(list(idx_test)).cuda())
# preval_embs = embeds[0, idx_val]
# test_embs = embeds[0, idx_test]
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_adj = adj.todense().A
# test_adj = test_adj[:,idx_test][idx_test,:].A
testneighbors = [[]for m in range(len(idx_test)) ]
testneighbors_2hop = [[]for m in range(len(idx_test))  ]
for step,x in enumerate(idx_test):
    tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_test[step])
    testneighbors[step] = tempneighbors
    testneighbors_2hop[step] = tempneighbors_2hop

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
        idx_train = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/idx.pt").type(torch.long).cuda()
        # print(len(idx_train))
        neighbors = [[]for m in range(len(idx_train)) ]
        neighbors_2hop = [[]for m in range(len(idx_train))  ]
        # print(neighbors)
        # train_adj = adj.todense().A
        # train_adj = train_adj[:,train_range][train_range,:].A
        for step,x in enumerate(idx_train):
            tempneighbors,tempneighbors_2hop = process.find_2hop_neighbors(origin_adj,idx_train[step].item())
            neighbors[step] = tempneighbors
            neighbors_2hop[step] = tempneighbors_2hop
            # tempneighborsembs = embeds[0,tempneighbors]
            # tempneighbors_2hopembs = embeds[0,tempneighbors_2hop]

            # print(neighbors[step],neighbors_2hop[step])
            # print(tempneighborsembs)
        pretrain_embs = torch.index_select(embeds,0,torch.tensor(list(idx_train)).cuda())
        train_lbls = torch.load(f"data/fewshot_{args.dataset}/{shotnum}-shot_{args.dataset}/{i}/labels.pt").type(torch.long).squeeze().cuda()
        print("true",i,train_lbls)

        log = downprompt(neighbors, neighbors_2hop, nb_classes, embeds, train_lbls,
                         hop_level=args.hop_level, reduction=args.reduction, multi_prompt=args.multi_prompt,
                         bottleneck_size=args.bottleneck_size,
                         meta_in=args.meta_in,  out_size=args.out_size,
                         hidden_size=hid_units, prompt=args.prompt, use_metanet=args.use_metanet)

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
            logits=log.forward(train=1, idx=idx_train).float().cuda()
            loss = xent(logits, train_lbls)
            if not loss.requires_grad:
                loss.requires_grad = True

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
            logits = log.forward(train=0, neighbors=testneighbors, neighbors_2hop=testneighbors_2hop,
                                 idx=torch.tensor(idx_test).cuda())
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
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
            [accs.mean().item()], [accs.std().item()], [shotnum])),
        columns=['model_name', 'pretrained_model', 'hop_level', 'aug_type','drop_percent','multi_prompt',
                 'prompt', 'metanet', 'bottleneck','accuracy', 'mean', 'shotnum'])
    if os.path.exists(f"data/{args.dataset}/{args.filename}.csv"):
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv", mode='a', header=False)
    else:
        df.to_csv(f"data/{args.dataset}/{args.filename}.csv")