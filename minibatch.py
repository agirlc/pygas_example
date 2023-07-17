import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import ClusterData, ClusterLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from tqdm import tqdm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

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
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x.to(args.device), batch.adj_t.to(args.device))[:batch.batch_size]
        y = batch.y.squeeze(1)[:batch.batch_size].to(args.device)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * batch.batch_size
        total_count += batch.batch_size
        loss.backward()
        optimizer.step()

    return total_loss / total_count


@torch.no_grad()
def test(args, model, data, loader, evaluator):
    model.eval()
    outs = []
    for batch in loader:
        batch = batch.to(args.device)
        pred = model(batch.x, batch.adj_t)
        outs.append(pred[:batch.batch_size].cpu())
    out = torch.cat(outs, dim=0)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    
    loss = F.cross_entropy(out[data.val_idx], data.y.squeeze(1)[data.val_idx])
    train_acc = evaluator.eval({
        'y_true': data.y[data.train_idx],
        'y_pred': y_pred[data.train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_idx],
        'y_pred': y_pred[data.val_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_idx],
        'y_pred': y_pred[data.test_idx],
    })['acc']

    return loss.item(), train_acc, valid_acc, test_acc


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

    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=False), ])
    dataset = PygNodePropPredDataset(root=args.root, name='ogbn-arxiv', transform=transform)

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()
    data.train_idx = split_idx['train']
    data.val_idx = split_idx['valid']
    data.test_idx = split_idx['test']
    
    num_neighbors = [-1] * args.num_layers
    if args.sampling:
        num_neighbors = [20, 10, 5, 3, 1][:args.num_layers]
        
    kwargs = {'batch_size': 1024, 'num_workers': 4, 'persistent_workers': True}

    train_loader = NeighborLoader(
        data, 
        input_nodes=data.train_idx,
        num_neighbors=num_neighbors,
        shuffle=True,
        **kwargs
    )
    loader = NeighborLoader(
        data, 
        num_neighbors=[-1]*args.num_layers,
        shuffle=False,
        **kwargs
    )

   
    model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

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
            result = test(args, model, data, loader, evaluator)

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
