import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_sparse
from torch import FloatTensor
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

class H2GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True,
            # sample=None,
    ):
        super(H2GCN, self).__init__()
        # self.sample = torch.tensor(sample, dtype=int).cuda()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_out = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)
        nn.init.xavier_uniform_(self.w_out)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward_pretrain(self, adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        print(x.shape, self.w_embed.data.shape)
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        logits = torch.mm(r_final, self.w_out)
        # lploss = compareloss(logits, self.sample, temperature=10)
        return logits
    def embed(self,seq,adj,train_graph_nodes=None,graph_len=None):
        self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(seq, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        if train_graph_nodes !=None:
            logits_ = []
            for i in train_graph_nodes:
                logits_.append(torch.sum(r_final.unsqueeze(0)[0, i], dim=0))
            logits_graph = torch.stack(logits_, dim=0)
        if graph_len != None:
            logits_graph = split_and_batchify_graph_feats(r_final, graph_len)
        logits = torch.mm(logits_graph, self.w_classify)
        return logits
    def forward(self,seq,adj):
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(seq, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        logits = torch.mm(r_final, self.w_classify)
        return logits
    def forward1(self,seq,adj,train_graph_nodes=None,graph_len=None):
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(seq, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        if train_graph_nodes !=None:
            logits_ = []
            for i in train_graph_nodes:
                logits_.append(torch.sum(r_final.unsqueeze(0)[0, i], dim=0))
            logits_graph = torch.stack(logits_, dim=0)
        if graph_len != None:
            logits_graph = split_and_batchify_graph_feats(r_final, graph_len)
        logits = torch.mm(logits_graph, self.w_classify)
        return logits





def mygather(feature, index):
    # print("index",index)
    # print("indexsize",index.shape)
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):

    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

