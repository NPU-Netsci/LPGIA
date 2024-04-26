import abc
import time
import os

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import utils

from copy import deepcopy

from tqdm import tqdm


"""
gnn par
in hidden out
act
bn or ln
conv spc
model spc

gcn 
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=True, add_self_loops=True, conv_cached=False
sgc
in_channels: int, k: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 add_self_loops=True, conv_cached=False
mlp
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False
gat
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int, heads: Union[int, Arrays],
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 add_self_loops=True, att_concat=True, ns=0.2, att_dropout: float = 0
graphsage
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=False, conv_aggr='mean', conv_root_weight=True, conv_project=False
gcnguard
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.0, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=True, add_self_loops=True, conv_cached=False
simpgcn                 
in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=True, add_self_loops=True, conv_cached=False,
                 knn_cached=True, knn_cached_path='./simpgcn/knn_cache/', gamma=0.1, ssl_lambda=5, bias_init=0                 
                 
other par
lp

"""


def fit_gnn(gnn, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor, train_mask, val_mask=None, device='cpu',
            initialize=True, train_iters=200, patience=np.inf, lr=0.01, wd=5e-4,
            verbose=True, save=False, save_step=50, save_path=None):

    if initialize:
        gnn.reset_parameters()
    # if save:
    #     save_path = save_path + time.strftime("%Y%m%d%H%M", time.localtime()) + r'/'
    #     print('will be save gnn model in:%s' % save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)

    # x = x.detach().clone()
    # edge_index = edge_index.detach().clone()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    gnn.to(device)

    optimizer = optim.Adam(gnn.parameters(), lr=lr, weight_decay=wd)
    best_acc_val = 0
    # best_loss_val = 0
    best_weights = gnn.state_dict()
    best_output = None
    patience_waste = 0

    epochs = range(train_iters)
    if verbose:
        epochs = tqdm(range(train_iters), desc=str(gnn.__class__.__name__) + ' training')
    for epoch in epochs:
        gnn.train()
        optimizer.zero_grad()
        output = gnn.forward(x, edge_index)
        loss_train = F.nll_loss(output[train_mask], y[train_mask])
        if hasattr(gnn, 'add_loss'):
            loss_train = loss_train + gnn.add_loss
        acc_train = utils.accuracy(output[train_mask], y[train_mask]).item()
        loss_train.backward()
        optimizer.step()
        # scheduler.step()
        if val_mask is not None:
            gnn.eval()
            with torch.no_grad():
                output = gnn.forward(x, edge_index)
                loss_val = F.nll_loss(output[val_mask], y[val_mask]).item()
            acc_val = utils.accuracy(output[val_mask], y[val_mask]).item()
            if verbose:
                epochs.set_postfix(acc_train=acc_train, loss_train=loss_train.item(), acc_val=acc_val,
                                   loss_val=loss_val)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_output = output
                best_weights = deepcopy(gnn.state_dict())
                patience_waste = 0
            else:
                patience_waste += 1
            # if loss_val > best_loss_val:
            #     best_loss_val = loss_val
            #     best_output = output
            #     best_weights = deepcopy(gnn.state_dict())
            #     patience_waste = 0
            #     print('save')
            # if acc_val < best_acc_val and loss_val < best_loss_val:
            #     patience_waste += 1
        if save and epoch % save_step == 0:
            torch.save(gnn.state_dict(), save_path+'%s_%d.pt' % (gnn.__class__.__name__, epoch))

        # 相对提前终止
        if type(patience) == list:
            if patience_waste > patience[0]:
                break
        # 绝对提前终止
        elif epoch > patience:
            break

    if val_mask is not None:
        gnn.load_state_dict(best_weights)
        if save:
            torch.save(gnn.state_dict(), save_path + '%s_best.pt' % gnn.__class__.__name__)
    else:
        best_output = output

    return best_output


@torch.no_grad()
def predict_gnn(gnn, x, edge_index, device='cpu'):
    gnn.eval()
    gnn.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    output = gnn.forward(x, edge_index)
    return output


@torch.no_grad()
def test_gnn(gnn, x, edge_index, y, test_mask=None, device='cpu', verbose=True):
    gnn.eval()
    gnn.to(device)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    output = gnn.forward(x, edge_index)

    if test_mask is None:
        test_mask = np.arange(y.shape[0])
    loss = F.nll_loss(output[test_mask], y[test_mask]).item()
    acc = utils.accuracy(output[test_mask], y[test_mask]).item()
    if verbose:
        print(gnn.__class__.__name__, 'loss:', loss, 'acc:', acc)
    return output, loss, acc

