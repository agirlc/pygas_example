import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from tqdm import tqdm
import time

from torch_geometric_autoscale import ScalableGNN
from torch_geometric_autoscale import metis, permute, SubgraphLoader, EvalSubgraphLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCN(ScalableGNN):                 #++++++++++++++++++++++++++++++++++
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, buffer_size=5000):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size=2, buffer_size=buffer_size)             #++++++++++++++++++++++++++++++++++
        self.out_channels = out_channels     #++++++++++++++++++++++++++++++++++
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, *args):                                      #++++++++++++++++++++++++++++++++++
        reg = 0
        for i, conv in enumerate(self.convs[:-1]):
            if i > 0 and self.training:                                              #++++++++++++++++++++++++++++++++++
                approx = conv.lin(x+0.1 * torch.randn_like(x))     #++++++++++++++++++++++++++++++++++
                real = conv.lin(x)
                diff = (real - approx).norm(dim=-1)                          #++++++++++++++++++++++++++++++++++
                reg += diff.mean() / len(self.histories)                    #++++++++++++++++++++++++++++++++++
            x = conv(x, adj_t)
                
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.push_and_pull(self.histories[i], x, *args)        #++++++++++++++++++++++++++++++++++
        x = self.convs[-1](x, adj_t)
        if self.training:
            return x, reg
        else:
            return x
    
    #++++++++++++++++++++++++++++++++++
    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        h = self.convs[layer](x, adj_t)

        if layer < self.num_layers - 1:
            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h
    #++++++++++++++++++++++++++++++++++


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x#.log_softmax(dim=-1)

    
def train(args, model, loader, optimizer):
    model.train()

    total_loss = total_count = 0
    for (batch, *params) in loader:
        optimizer.zero_grad()
        bs = params[0]
        out, reg = model(batch.x.to(args.device), batch.adj_t.to(args.device), *params)[:bs]
        y = batch.y.squeeze(1)[:bs].to(args.device)
        mask = batch.train_mask[:bs].to(model.device)
        loss = F.cross_entropy(out[mask], y[mask]) + 0.1 * reg
        total_loss += loss.item() * bs
        total_count += bs
        loss.backward()
        optimizer.step()

    return total_loss / total_count


@torch.no_grad()
def test(args, model, data, loader, evaluator, y_shape):
    model.eval()
    outs = model(loader=loader)
    # outs = torch.zeros(y_shape)
    # for (batch, *params) in loader:
    #     bs = params[0]
    #     n_id = params[1]
    #     out = model(batch.x.to(args.device), batch.adj_t.to(args.device), *params)[:bs]
    #     outs[n_id[:bs]] = out.cpu()
    y_pred = outs.argmax(dim=-1, keepdim=True)
    
    
    loss = F.cross_entropy(outs[data.val_mask], data.y.squeeze(1)[data.val_mask])
    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']

    return loss.item(), train_acc, valid_acc, test_acc


def make_mask(data, idx):
    mask = torch.zeros(data.size(0), dtype=torch.bool)
    mask[idx] = True
    return mask

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--root', type=str, default='/home/qqg/data/')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--sampling', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #++++++++++++++++++++++++++++++++++
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=False), ])
    dataset = PygNodePropPredDataset(root=args.root, name='ogbn-arxiv', transform=transform)

    data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.train_mask = make_mask(data, split_idx['train'])
    data.val_mask = make_mask(data, split_idx['valid'])
    data.test_mask = make_mask(data, split_idx['test'])
    
    data.adj_t = gcn_norm(data.adj_t, add_self_loops=True)    #++++++++++++++++++++++++++++++++++
    perm, ptr = metis(data.adj_t, num_parts=1000, log=True)   # len(data) = 169343, 169343 / 128 ~~ 1000
    data = permute(data, perm, log=True)
    train_loader = SubgraphLoader(data, ptr, batch_size=10, shuffle=True, num_workers=4, persistent_workers=True )  # batch size is subgraph count!
    eval_loader = EvalSubgraphLoader(data, ptr, batch_size=10)
    
    t = time.perf_counter()
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader]) * 10
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')
   
    model = GCN(data.size(0), data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, buffer_size=buffer_size).to(device)
    #++++++++++++++++++++++++++++++++++
    
    
    evaluator = Evaluator(name='ogbn-arxiv')

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val_acc = 0
        best_test_acc = 0
        best_loss = float('inf')
        wandering = 0
        pbar = tqdm(total=args.epochs)
        for epoch in range(1, 1 + args.epochs):
            loss = train(args, model, train_loader, optimizer)
            result = test(args, model, data, eval_loader, evaluator, y_shape=(data.size(0), dataset.num_classes))

            val_loss, train_acc, val_acc, test_acc = result
            if val_loss < best_loss:
                best_loss = val_loss
                best_val_acc = val_acc
                best_test_acc = test_acc
                wandering = 0
            else:
                wandering += 1
                
            out = f'{epoch:02d}: Loss={loss:.4f}, acc={100*train_acc:.2f}%/{100*val_acc:.2f}%/{100*test_acc:.2f}%'
            out += f', best={100*best_val_acc:.2f}%/{100*best_test_acc:.2f}%'
            pbar.set_description(out)
            pbar.update(1)

        pbar.close()
        usage = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
        print(f'Max gpu memory usage: {usage:.2f} MB')


if __name__ == "__main__":
    main()
