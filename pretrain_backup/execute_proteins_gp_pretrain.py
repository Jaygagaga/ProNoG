import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import scipy.sparse as sp

import random
# from models import LogReg
from preprompt import PrePrompt
import preprompt
from utils import process

import aug
import os
import tqdm
import argparse
from downprompt import downprompt
import csv

from tqdm import tqdm
parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="PROTEINS", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='modelset/model_PROTEINS1_64.pkl', help='save ckpt name')
parser.add_argument('--local_rank', type=str, help='local rank for dist')
args = parser.parse_args()


print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
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
nb_epochs = 5
test_times = 50
#test_times = 10
patience = 10
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 64
sparse = False
useMLP =False
class_num = 3
LP = False


nonlinearity = 'prelu'  # special name to separate parameters


if args.dataset == 'PROTEINS':
    dataset = TUDataset(root='data', name=args.dataset,use_node_attr=True)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)

# test_adj = torch.load(f"data/fewshot_{args.dataset}/5-shot_{args.dataset}/testadj.pt").squeeze(0)
# test_adj_ = test_adj.numpy()
# adj_csr = sp.csr_matrix(test_adj_)
# test_adj = process.normalize_adj(adj_csr + sp.eye(adj_csr.shape[0]))
# sp_adj_test = process.sparse_mx_to_torch_sparse_tensor(test_adj).cuda()


a1 = 0.9    #dgi
a2 = 0.9    #graphcl
a3 = 0.1    #lp
ft_size = 1

device = torch.device("cuda")
model = PrePrompt(ft_size, hid_units, nonlinearity,a1,a2,a3,1,0.3)
model = model.to(device)

best = 1e9

for epoch in range(nb_epochs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    regloss = 0
    for step, data in enumerate(loader):
        print(step)
        features,adj,nodelabels= process.process_tu(data,ft_size)

        negetive_sample = preprompt.prompt_pretrain_sample(adj,100)


        nb_nodes = features.shape[0]  # node number
        # ft_size = features.shape[1]  # node features dim
        nb_classes = nodelabels.shape[1]  # classes = 6
        if args.use_origin_feature == 0:
            features = process.preprocess_features(features)
        features = torch.FloatTensor(features[np.newaxis])

        '''
        # ------------------------------------------------------------
        # edge node mask subgraph
        # ------------------------------------------------------------
        '''
        # print("Begin Aug:[{}]".format(args.aug_type))
        # if args.aug_type == 'edge':

        aug_features1edge = features
        aug_features2edge = features

        aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
        aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges

        '''
        # ------------------------------------------------------------
        '''

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
        aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
            sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)


        else:
            adj = adj.todense()
            aug_adj1edge = aug_adj1edge .todense()
            aug_adj2edge = aug_adj2edge .todense()

        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
            aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])
            # aug_adj1mask = torch.FloatTensor(aug_adj1mask[np.newaxis])
            # aug_adj2mask = torch.FloatTensor(aug_adj2mask[np.newaxis])
        labels = torch.FloatTensor(nodelabels[np.newaxis])

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available() and step==0:
            print('Using CUDA')
            # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
            features = features.cuda()
            aug_features1edge = aug_features1edge.cuda()
            aug_features2edge = aug_features2edge.cuda()
            # aug_features1mask = aug_features1mask.cuda()
            # aug_features2mask = aug_features2mask.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
                sp_aug_adj1edge = sp_aug_adj1edge.cuda()
                sp_aug_adj2edge = sp_aug_adj2edge.cuda()
                # sp_aug_adj1mask = sp_aug_adj1mask.cuda()
                # sp_aug_adj2mask = sp_aug_adj2mask.cuda()
            else:
                adj = adj.cuda()
                aug_adj1edge = aug_adj1edge.cuda()
                aug_adj2edge = aug_adj2edge.cuda()
                # aug_adj1mask = aug_adj1mask.cuda()
                # aug_adj2mask = aug_adj2mask.cuda()
            labels = labels.cuda()
            # idx_train = idx_train.cuda()
            # idx_val = idx_val.cuda()
            # idx_test = idx_test.cuda()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        best = 1e9

        model.train()
        optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logit = model(features, shuf_fts, aug_features1edge, aug_features2edge,
                       sp_adj if sparse else adj,
                    sp_aug_adj1edge if sparse else aug_adj1edge,
                    sp_aug_adj2edge if sparse else aug_adj2edge,
                    sparse, None, None, None, lbl=lbl,sample=negetive_sample)
        loss = loss + logit
        showloss = loss/(step+1)

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
#



model.load_state_dict(torch.load(args.save_name))
dgiprompt = model.dgi.prompt
graphcledgeprompt = model.graphcledge.prompt
lpprompt = model.lp.prompt
downstreamlr = 0.01

# if not os.path.exists(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors.csv"):
#     testneighbors = [[] for m in range(len(idx_test))]
#     testneighbors_2hop = [[] for m in range(len(idx_test))]
#     for step, x in enumerate(idx_test):
#         tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(origin_adj, idx_test[step], k=args.k)
#         testneighbors[step] = tempneighbors
#         testneighbors_2hop[step] = tempneighbors_2hop
#     with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors1.csv", "w") as f:
#         wr = csv.writer(f)
#         wr.writerows(testneighbors)
#     with open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors_2hop1.csv", "w") as f:
#         wr = csv.writer(f)
#         wr.writerows(testneighbors_2hop)
# else:
#     testneighbors = []
#     file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors1.csv")
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         testneighbors.append([int(i) for i in row])
#
#     testneighbors_2hop = []
#     file = open(f"/home/xingtong/WebKB_node_origin/data/fewshot_{args.dataset}/testneighbors_2hop1.csv")
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         testneighbors_2hop.append([int(i) for i in row])

for shotnum in range(2,11):
    print("shotnum",shotnum)

    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    test_adj = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/5-shot_PROTEINS/testadj.pt").squeeze().cuda()
    print('-' * 100)
    # print("test_adj",test_adj.shape)
    testfeature = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/5-shot_PROTEINS/testemb.pt").cuda()
    # print("testfeature",testfeature.shape)
    test_embs , _= model.embed(testfeature,test_adj, sparse, None,LP)
    # print("testemb1",test_embs.shape)
    test_embs =test_embs.squeeze()
    # print("testemb2",test_embs)
    test_lbls = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/5-shot_PROTEINS/testlabels.pt").type(torch.long).squeeze().cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    print('-' * 100)
    # print("test_adj",test_adj.shape)
    # print("testfeature",testfeature.shape)
    print('-' * 100)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    nb_classes = 3

    for i in range(100):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print("tasknum",i)
        pretrain_adj = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/{}-shot_PROTEINS/{}/nodeadj.pt".format(shotnum,i)).squeeze().cuda()
        prefeature = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/{}-shot_PROTEINS/{}/nodeemb.pt".format(shotnum,i)).cuda()
        if args.use_origin_feature == 0:
            prefeature = process.preprocess_features(prefeature)
        print("feature:",prefeature.shape)
        pretrain_embs , _= model.embed(prefeature,pretrain_adj, sparse, None,LP)
        pretrain_embs = pretrain_embs.squeeze()
        print("embs:",pretrain_embs.shape)
        pretrain_lbls = torch.load("/home/xingtong/WebKB_node_origin/data/fewshot_PROTEINS/{}-shot_PROTEINS/{}/nodelabels.pt".format(shotnum,i)).type(torch.long).squeeze().cuda()
        pretrain_labels = torch.zeros(pretrain_lbls.shape[0],nb_classes).cuda()
        for j in range(pretrain_lbls.shape[0]):
            pretrain_labels[j][pretrain_lbls[j]]=1
        print("true",pretrain_lbls)

        #sub_graph_embs = torch.sum(pretrain_embs, dim=0, keepdim=True)
        #log = downprompt(dgiprompt, graphcledgeprompt, lpprompt, hid_units, nb_classes,sub_graph_embs,pretrain_lbls)

        log = downprompt(dgiprompt, graphcledgeprompt, lpprompt, hid_units, nb_classes,pretrain_embs,pretrain_lbls)
        # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
        opt = torch.optim.Adam(log.parameters(), lr=0.001)
        log.cuda()
        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(50):
            log.train()
            opt.zero_grad()
            logits = log(pretrain_embs,1).float().cuda()
            #logits = log(sub_graph_embs,1).float().cuda()
            loss = xent(logits, pretrain_labels)
            loss.backward()
            opt.step()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print('acc:[{:.4f}]'.format(acc))
        tot += acc
    
    print('-' * 100)
    print("shotnum",shotnum)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / test_times))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)
    row = [args.save_name,shotnum,downstreamlr,accs.mean().item(),accs.std().item()]
    out = open("data/fewshot_PROTEINS.csv", "a", newline="")
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(row)
print("test_lablels",test_lbls)
