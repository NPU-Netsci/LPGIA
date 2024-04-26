

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import utils

from typing import Union, Optional, Callable, Optional

from tqdm import tqdm
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from utils import Arrays


class LabelPropagation(MessagePassing):
    """The label propagation operator from the `"Learning from Labeled and
    Unlabeled Data with Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Args:
        num_layers (int): The number of propagations.
        alpha (float): The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers: int, alpha: float = 0.9,
                 memory=False, add_self_loops=False, auto_convergence=False):
        super().__init__(aggr='add')
        self.nlayers = num_layers
        self.alpha = alpha
        self.add_self_loops = add_self_loops

        self.memory = True if auto_convergence else memory
        self.out_memory = []
        self.auto_convergence = auto_convergence

    def reset_parameters(self):
        pass

    def forward(self, y: Tensor, edge_index, mask: Optional[Tensor] = None, edge_weight: Optional[Tensor] = None,
                post_step: Callable = lambda y: y.clamp_(0., 1.)) -> Tensor:
        """"""
        # 清空缓存
        self.out_memory = []

        if y.dtype == torch.long and y.size(0) == y.numel():
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=self.add_self_loops)

        res = (1 - self.alpha) * out
        for i in range(self.nlayers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            out.mul_(self.alpha).add_(res)
            out = post_step(out)

            if self.memory:
                self.out_memory.append(out)
            if self.auto_convergence and i > 0:
                if np.abs(self.out_memory[-1].sum()-self.out_memory[-2].sum()) < 1e-6:
                    print(f'label propagation auto convergence at epoch: {i-1}')
                    break

        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_layers={self.nlayers}, '
                f'alpha={self.alpha})')


@torch.no_grad()
def test_lp(lpm, y, edge_index, mask=None, edge_weight=None, test_mask=None,
            post_step: Callable = lambda y: y.clamp_(0., 1.), device='cpu', verbose=True):
    lpm.eval()
    lpm.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    y = y.to(device)

    output = lpm.forward(y, edge_index, mask, edge_weight, post_step)

    if test_mask is None:
        test_mask = np.arange(y.shape[0])
    loss = F.nll_loss(output[test_mask], y[test_mask]).item()
    acc = utils.accuracy(output[test_mask], y[test_mask]).item()
    if verbose:
        print(lpm.__class__.__name__, 'loss:', loss, 'acc:', acc)

    return output, loss, acc
