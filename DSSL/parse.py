from models import LINK, GCN, MLP, SGC, GAT, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net, LINK_Concat, LINKX, GPRGNN, GCNII, GCN_Net, GCN_G3NN
from data_utils import normalize

def parse_method(args, data, n, c, d, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
    elif args.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=not args.no_bn).to(device)
    elif args.method == 'gcn-net':
        model = GCN_Net(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    dropout=args.dropout).to(device)
    elif args.method == 'gcn-g3nn':
        model = GCN_G3NN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    dropout=args.dropout).to(device)
    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha, num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha, dropout=args.dropout, num_layers=args.num_layers).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, heads=args.gat_heads).to(device)
    elif args.method == 'lp':
        mult_bin = args.dataset=='ogbn-proteins'
        model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
    elif args.method == 'h2gcn':
        model = H2GCN(d, args.hidden_channels, c, data.edge_index,
                        data.num_nodes,
                        num_layers=args.num_layers, dropout=args.dropout,
                        num_mlp_layers=args.num_mlp_layers).to(device)
    elif args.method == 'link_concat':
        model = LINK_Concat(d, args.hidden_channels, c, args.num_layers, data.num_nodes, dropout=args.dropout).to(device)
    elif args.method == 'linkx':
        model = LINKX(d, args.hidden_channels, c, args.num_layers, data.num_nodes,
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'gcn2':
        model = GCNII(d, args.hidden_channels, c, args.num_layers, args.gcn2_alpha, args.theta, dropout=args.dropout).to(device)
    else:
        raise ValueError('Invalid method')
    return model
