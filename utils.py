import copy
import os.path as osp
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
import torch_geometric.transforms as T

from typing import Union, Optional

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from sklearn.model_selection import train_test_split


Arrays = Union[tuple, list, torch.Tensor, np.ndarray]


def accuracy(output, labels):
    """
    compute accuracy
    :param output: 2d tensor
    :param labels: 1d ndarray or tensor
    :return: acc
    """

    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def normalize_adj(adj, add_self_loop=True):
    mx = copy.deepcopy(adj)
    if type(mx) in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix, sp.lil_matrix]:
        if type(mx) is not sp.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 and add_self_loop:
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1 / 2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)
        return mx
    elif type(mx) == torch_sparse.SparseTensor:
        mx = transform_sparse_mat_type(mx, 'csr')
        mx = normalize_adj(mx, add_self_loop)
        return transform_sparse_mat_type(mx, 'adj_t')
    else:
        raise NotImplementedError(type(adj))


def transform_sparse_mat_type(src, dst_type, t_operation=True):
    """
    transform sparse matrix from src type to dst type, support type including csr, edge_index, adj_t, tensor

    notice that torch_sparse.SparseTensor has T operation

    :param t_operation:
    :param src: ori adj
    :param dst_type: target adj type
    :return:
    """
    row = None  # 1d ndarray
    col = None  # 1d ndarray
    data = None
    size = None

    # scipy adj should avoid use too much type
    if type(src) in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix, sp.lil_matrix]:
        src = src.tocoo()
        row = src.row
        col = src.col
        data = src.data
        size = src.shape
    elif type(src) == torch.Tensor and not src.is_sparse and src.shape[0] == 2:
        row = src[0, :].cpu().numpy()
        col = src[1, :].cpu().numpy()
        data = np.ones_like(row)
        size = (np.max(row)+1, np.max(col)+1)
    elif type(src) == torch.Tensor and src.is_sparse:

        row = src.indices()[0, :].cpu().numpy()
        col = src.indices()[1, :].cpu().numpy()
        data = src.values().cpu().numpy()
        size = (src.shape[0], src.shape[1])
    elif type(src) == torch_sparse.SparseTensor:
        if t_operation:
            row, col, data = src.t().coo()
        else:
            row, col, data = src.coo()
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        data = data.cpu().numpy()
        size = src.sparse_size()
    else:
        raise ValueError(type(src))

    if dst_type == 'csr':
        return sp.coo_matrix((data, (row, col)), shape=size).tocsr()
    elif dst_type == 'edge_index':
        return torch.vstack((torch.LongTensor(row), torch.LongTensor(col)))
    elif dst_type == 'adj_t':
        if t_operation:
            return torch_sparse.SparseTensor(row=torch.LongTensor(col), col=torch.LongTensor(row),
                                             value=torch.FloatTensor(data), sparse_sizes=size)  # t operation
        else:
            return torch_sparse.SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col),
                                             value=torch.FloatTensor(data), sparse_sizes=size)
    elif dst_type == 'tensor':
        return torch.sparse.FloatTensor(torch.vstack((torch.LongTensor(row), torch.LongTensor(col))),
                                        torch.FloatTensor(data), torch.Size(size)).coalesce()
    else:
        raise ValueError(dst_type)


def reshape_csr(mx, shape):
    indices = mx.tocsr(copy=True).nonzero()
    return sp.csr_matrix((mx.tocsr().data, (indices[0], indices[1])), shape=shape)


class GraphDataset:

    def __init__(self, name: str, path: str, split='nettack', is_lcc=True, seed: int = None, verbose=False):
        # basic components
        self.name = name
        self.adj = None  # csr matrix
        self.edge_index = None  # 2d tensor
        # self.adj_t = None  # torch_sparse.SparseTensor
        self.features = None  # 2d ndarray
        # self.x = None  # 2d tensor
        self.labels = None  # 1d ndarray
        # self.y = None  # 1d tensor
        self.idx_train = None  # 1d ndarray
        self.idx_val = None  # 1d ndarray
        self.idx_test = None  # 1d ndarray
        self.nnodes = 0
        self.nedges = 0
        self.nfeats = 0
        self.nclass = 0
        # config
        self.is_lcc = is_lcc  # only retain largest connected component
        self.split = split  # split nodes that follow setting
        self.seed = seed

        # load data
        print(f'Loading dataset {name}')
        self.support_dataset_list = ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed', 'arxiv']
        if name not in self.support_dataset_list:
            raise NotImplementedError(name)
        self.load_data(name, path, verbose)

    def load_data(self, name, path, verbose):
        """
        load dataset including 'cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed', 'arxiv'.

        :param name: dataset name
        :param path: dataset dir path
        :param verbose: report dataset statistic
        :return:
        """
        # check config

        # load dataset
        if name in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed']:
            self.load_tiny_dataset(name, path)
        elif name in ['arxiv']:
            self.load_ogb_dataset(name, path)
        else:
            raise NotImplementedError(name)

        # data statistic
        self.name = name
        self.nnodes = self.adj.shape[0]
        self.nedges = self.adj.sum()/2
        self.nfeats = self.features.shape[1]
        self.nclass = self.labels.max() + 1

        # report info
        print(f'----* {self.name} info *----')
        print(f'adj shape: {self.adj.shape} || edge num: {self.nedges}')
        print(f'features shape: {self.features.shape}')
        print(f'labels shape: {self.labels.shape} || class num: {self.nclass}')
        print(f'train/val/test: {len(self.idx_train)}/{len(self.idx_val)}/{len(self.idx_test)}')
        if verbose:
            # avg degree
            degrees = np.array(self.adj.sum(1)).flatten()
            print(f'avg degrees: {np.mean(degrees)}')
            # avg non-zero feat
            average_feature_nums = np.diff(sp.csr_matrix(self.features).indptr).mean()
            zero_one_features = self.features.copy()
            zero_one_features[zero_one_features != 0] = 1
            if (zero_one_features - self.features).sum() == 0:
                print(f'binary feature || avg non-zero feature num: {average_feature_nums:.2f}/{self.nfeats}')
            else:
                print(f'continuous feature || avg non-zero feature num: {average_feature_nums:.2f}/{self.nfeats}')
        print(f'----* Finish Loading *----')

    def load_tiny_dataset(self, name, path):
        """
        load dataset including 'cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed'.

        :param name: dataset name
        :param path: dataset dir path
        :return:
        """
        # check config

        # load path
        path = osp.join(path, f'{name}.npz')
        npz_data = np.load(path, allow_pickle=True)
        # load adj
        adj = sp.csr_matrix((npz_data['adj_data'], npz_data['adj_indices'], npz_data['adj_indptr']),
                            shape=npz_data['adj_shape'])
        # load feature, notes that can be binary or continue
        if name in ['polblogs']:
            # 填充one-hot
            features = sp.eye(adj.shape[0], adj.shape[1]).tocsr()
        else:
            features = sp.csr_matrix((npz_data['attr_data'], npz_data['attr_indices'], npz_data['attr_indptr']),
                                     shape=npz_data['attr_shape'])
        # load label
        labels = npz_data['labels']

        # symmetric adj
        adj = adj + adj.T
        adj[adj > 1] = 1
        # lcc
        if self.is_lcc:
            print(f'Before lcc nodes: {adj.shape[0]} || edges: {adj.sum()/2}')
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, 'Graph contains singleton nodes'
        adj.setdiag(0)
        adj.eliminate_zeros()
        # check adj
        assert np.abs(adj-adj.T).sum() == 0, 'Graph is not symmetric'
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, 'Graph must be unweighted'

        # save data
        self.adj = adj
        self.edge_index = transform_sparse_mat_type(adj, 'edge_index')
        self.features = features.A
        self.labels = labels
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test(labels, self.split, self.seed)  # split train/val/test

    def load_ogb_dataset(self, name, path):
        """
        load dataset including 'cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed'.

        :param name: dataset name
        :param path: dataset dir path
        :return:
        """
        # check config

        # load path
        # pyg_dataset = PygNodePropPredDataset(f'ogbn-{name}', path, transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))  # adj_t
        pyg_dataset = PygNodePropPredDataset(f'ogbn-{name}', path, transform=T.Compose([T.ToUndirected()]))  # edge_index

        # save data
        self.adj = transform_sparse_mat_type(pyg_dataset[0]['edge_index'], 'csr')
        self.edge_index = pyg_dataset[0]['edge_index']
        self.features = pyg_dataset[0]['x'].numpy()
        self.labels = pyg_dataset[0]['y'].numpy().flatten()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test(pyg_dataset, 'ogb', self.seed)  # use ogb split

        del pyg_dataset

    @staticmethod
    def largest_connected_components(adj, n_components=1):
        """
        Select k largest connected components.
        """
        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print(f'Selecting {n_components} largest connected components')
        return nodes_to_keep

    @staticmethod
    def get_train_val_test(labels, split, seed):
        """
        get train/val/test idx set.
        """

        if split == 'ogb' or type(labels) == PygNodePropPredDataset:
            split_idx_dict = labels.get_idx_split()
            idx_train = split_idx_dict['train'].numpy()
            idx_val = split_idx_dict['valid'].numpy()
            idx_test = split_idx_dict['test'].numpy()
            return idx_train, idx_val, idx_test

        if labels is not None:
            idx_num = labels.shape[0]
        else:
            raise RuntimeError('Try to split empty label set.')
        stratify = labels  # data is expected to split in a stratified fashion. So stratify should be labels.
        if seed is not None:
            np.random.seed(seed)

        idx = np.arange(idx_num)
        idx_train = None
        idx_val = None
        idx_test = None

        # nettack setting: we split the nodes into 10% training, 10% validation and 80% testing data
        if split == 'nettack':
            train_size = 0.1
            val_size = 0.1
            test_size = 0.8

            idx_train_and_val, idx_test = train_test_split(idx,
                                                           random_state=None,
                                                           train_size=train_size+val_size,
                                                           test_size=test_size,
                                                           stratify=stratify)
            stratify = stratify[idx_train_and_val]
            idx_train, idx_val = train_test_split(idx_train_and_val,
                                                  random_state=None,
                                                  train_size=(train_size / (train_size + val_size)),
                                                  test_size=(val_size / (train_size + val_size)),
                                                  stratify=stratify)
        # gcn setting: we randomly sample 20 instances for each class as training data, 500 instances as validation data, 1000 instances as test data.
        elif split == 'gcn':
            nclass = labels.max() + 1
            idx_train = []
            idx_unlabeled = []
            for i in range(nclass):
                labels_i = idx[labels == i]
                labels_i = np.random.permutation(labels_i)
                idx_train = np.hstack((idx_train, labels_i[: 20])).astype(int)
                idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20:])).astype(int)

            idx_unlabeled = np.random.permutation(idx_unlabeled)
            idx_val = idx_unlabeled[: 500]
            idx_test = idx_unlabeled[500: 1500]
        # list setting: we split the nodes according to the input list
        elif type(split) == list:
            if len(split) == 3:
                train_size = split[0]
                val_size = split[1]
                test_size = split[2]

                idx_train_and_val, idx_test = train_test_split(idx,
                                                               random_state=None,
                                                               train_size=train_size + val_size,
                                                               test_size=test_size,
                                                               stratify=stratify)
                stratify = stratify[idx_train_and_val]
                idx_train, idx_val = train_test_split(idx_train_and_val,
                                                      random_state=None,
                                                      train_size=(train_size / (train_size + val_size)),
                                                      test_size=(val_size / (train_size + val_size)),
                                                      stratify=stratify)

        return idx_train, idx_val, idx_test


def get_default_gnn_config(dataset_name, gnn_name):

    config = {}
    config['with_ln'] = False
    config['with_bn'] = False

    if dataset_name in ['cora', 'citeseer', 'cora_ml', 'pubmed']:

        config['n_layer'] = 2
        config['hidden_channels'] = 16
        config['dropout'] = 0.5

        if gnn_name == 'gat':
            config['n_head'] = 8
            config['hidden_channels'] = 8
        if gnn_name == 'simpgcn':
            config['knn_cached'] = True  #

    elif dataset_name in ['arxiv']:

        config['n_layer'] = 3  # 4
        config['hidden_channels'] = 256  # np.array([256, 128, 64]).astype(np.int)
        config['dropout'] = 0.5

        if gnn_name in ['gcn', 'sgc', 'graphsage', 'gat', 'gcnguard']:
            config['with_ln'] = True
        if gnn_name in ['gat']:
            config['n_head'] = 4
            config['hidden_channels'] = 64
        if gnn_name == 'simpgcn':
            config['knn_cached'] = True  #

    else:
        raise NotImplementedError(dataset_name)

    return config


def get_default_gia_config(dataset_name, gia_name):

    config = {}

    if gia_name == 'lpgia':
            config['a1'] = 0.5
            config['a2'] = 0.5
            config['opt_fake_feature'] = 'weight'
            config['opt_homophily_score'] = 'hs'
            config['opt_cluster'] = 'lp'
            config['feature_budget'] = None
            config['sparse_feature'] = True
            if dataset_name in ['arxiv']:
                config['sparse_feature'] = False
                config['opt_fake_feature'] = 'dense'
    else:
        raise NotImplementedError(dataset_name, gia_name)

    return config




