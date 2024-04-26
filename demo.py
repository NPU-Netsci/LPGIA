import argparse
import time
import os

import pandas as pd
import torch
import numpy as np
import random

from typing import Union, Optional

from torch_geometric.data import Data

import utils
from utils import GraphDataset
from gnnpanzer.gnn import *
from gnnpanzer.gia import *

parser = argparse.ArgumentParser(description='test attack method')
# normal attack setting
parser.add_argument('--atk_name', type=str, default='lpgia', help='attack_method')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--device', type=str, default=None, help='device')
parser.add_argument('--n_test', type=int, default=50, help='number of test')
parser.add_argument('--dataset_names', type=str, default='cora,cora_ml,citeseer', help='datasets')
parser.add_argument('--atk_config_path', type=str, default=None)
parser.add_argument('--save_result', type=bool, default=False, help='whether save result using excel')
parser.add_argument('--result_path', type=str, default='./results/')
# normal gnn setting
parser.add_argument('--gnn_names', type=str, default='gcn,gat', help='gnns')
parser.add_argument('--load_pretraining_gnn', type=bool, default=False, help='whether load pretraining model')
parser.add_argument('--pretraining_gnn_path', type=str, default='./pretraining_gnn/')
parser.add_argument('--save_gnn', type=bool, default=False, help='whether save graphs after perturbations')
parser.add_argument('--gnn_path', type=str, default='./pretraining_gnn/')
parser.add_argument('--gnn_config_path', type=str, default=None)
#
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = args.device


def gia_test(gia_name: str = 'lpgia', n_test: int = 50, dataset_names: Union[str, list] = None, gnn_names: Union[str, list] = None,
             load_pretraining_gnn=False, pretraining_gnn_path='./pretraining_gnn/', fine_tune=False,
             gia_config_path: str = None, gnn_config_path: str = None,
             load_cache = False, cache_path = './cache/',
             save_ptb_graphs=False, ptb_graph_path='./ptb_graphs/', load_ptb_graphs=False,
             save_result=False, result_path='./results/',
             ptb_rate: Union[int, float] = 0.05, evasion=True, poisoning=True, sp_a = 0.5,
             verbose=True):
    """黑盒设置下的攻击与验证，允许通过训练集标签构建替代模型"""

    if save_ptb_graphs and load_ptb_graphs:
        raise ValueError('Can not save and load ptb graph in the same task.')

    demo_timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
    res_df = pd.DataFrame(columns=['dataset', 'model', 'range', 'acc_eva', 'acc_poi'])

    if dataset_names is None:
        dataset_names = ['cora', 'cora_ml', 'citeseer', 'pubmed']  # full
        # dataset_names = ['cora', 'cora_ml', 'citeseer']  # tiny
        # single test
        # dataset_names = ['cora']
        # dataset_names = ['cora_ml']
        # dataset_names = ['citeseer']
        # dataset_names = ['pubmed']
        # dataset_names = ['arxiv']
    else:
        if type(dataset_names) == str:
            dataset_names = dataset_names.split(sep=',')


    if gnn_names is None:
        # gnn_names = ['gcn', 'sgc', 'gat', 'gcnguard', 'simpgcn']
        # single test
        gnn_names = ['gcn']
    else:
        if type(gnn_names) == str:
            gnn_names = gnn_names.split(sep=',')
    n_gnn = len(gnn_names)

    # report setting
    print('This GIA testing use attack model: %s repeat %d times, datasets including %s, victim models including %s' % (
          gia_name, n_test, dataset_names, gnn_names))
    print('The ptb rate or the number of fake nodes is %0.4f' % ptb_rate)
    print('Random seed:', args.seed)
    if load_pretraining_gnn:
        print('will be load pretraining model from: %s' % pretraining_gnn_path)
    if save_ptb_graphs:
        print('will be save perturbations in: %s' % ptb_graph_path)
    if load_ptb_graphs:
        print('will be load perturbations in: %s' % ptb_graph_path)
    if save_result:
        result_path = result_path + f'{demo_timestamp}/'
        print('will be save result in: %s' % result_path)

    res_df = pd.DataFrame(columns=['dataset', 'model', 'range', 'eva_acc', 'poi_acc'])

    # 逐数据集实验
    for dataset_name in dataset_names:
        # 读取数据集
        if dataset_name in ['cora', 'citeseer', 'cora_ml', 'pubmed', 'polblogs']:
            dataset = GraphDataset(dataset_name, './datasets/tiny/', seed=args.seed, verbose=True)
        elif dataset_name in ['arxiv']:
            dataset = GraphDataset(dataset_name, './datasets/ogb/', seed=args.seed, verbose=True)
        else:
            raise NotImplementedError(dataset_name)
        pyg_data = Data(x=torch.FloatTensor(dataset.features), edge_index=dataset.edge_index, y=torch.LongTensor(dataset.labels),
                        idx_train=dataset.idx_train, idx_val=dataset.idx_val, idx_test=dataset.idx_test)

        # init parameter
        # init num of fake node
        if ptb_rate is None:
            n_fake_nodes = int(dataset.adj.shape[0] * 0.05)
        elif ptb_rate <= 0:
            raise ValueError()
        elif ptb_rate <= 1 and type(ptb_rate) == float:
            n_fake_nodes = int(dataset.adj.shape[0] * ptb_rate)
        else:
            n_fake_nodes = int(ptb_rate)
        
        # preprocessing
        # 降低不确定性解决方案一：多次训练取平均
        if gia_name == 'lpgia':
            out_gcn_soft_avg = None
            weight_gcn_avg = None
            if load_cache:
                if dataset_name in ['cora', 'cora_ml', 'citeseer', 'pubmed', 'arxiv']:
                    out_gcn_soft_avg = torch.load(cache_path + "%s_%s_avgsoftout.pt" % ('gcn', dataset_name), map_location=device)
                    weight_gcn_avg = torch.load(cache_path + "%s_%s_avgweight.pt" % ('gcn', dataset_name), map_location=device)
                    dataset.idx_train = np.load(cache_path + "%s_%s_idx_train.npy" % (gia_name, dataset_name))
                    dataset.idx_val = np.load(cache_path + "%s_%s_idx_val.npy" % (gia_name, dataset_name))
                    dataset.idx_test = np.load(cache_path + "%s_%s_idx_test.npy" % (gia_name, dataset_name))
                    pyg_data.idx_train = dataset.idx_train
                    pyg_data.idx_val = dataset.idx_val
                    pyg_data.idx_test = dataset.idx_test
                else:
                    raise NotImplementedError()
            else:
                for i in range(50):
                    gcn = GCNModel(pyg_data.x.shape[1], 16, 2, int(pyg_data.y.max() + 1), dropout=0.5).to(device)
                    if dataset_name in ['arxiv']:
                        gcn = GCNModel(pyg_data.x.shape[1], 256, 3, int(pyg_data.y.max() + 1), dropout=0.5, with_ln=True).to(device)
                    output = fit_gnn(gcn, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                     train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
                    out_gcni_soft = torch.exp(output)
                    weight_gcni = gcn.get_linear_weight()
                    if out_gcn_soft_avg is None:
                        out_gcn_soft_avg = out_gcni_soft
                        weight_gcn_avg = weight_gcni
                    else:
                        out_gcn_soft_avg = out_gcn_soft_avg + out_gcni_soft
                        weight_gcn_avg = weight_gcn_avg + weight_gcni
                out_gcn_soft_avg = torch.mul(out_gcn_soft_avg, 1/100)
                weight_gcn_avg = torch.mul(weight_gcn_avg, 1/100)
                torch.save(out_gcn_soft_avg, cache_path + "%s_%s_avgsoftout.pt" % ('gcn', dataset_name))
                torch.save(weight_gcn_avg, cache_path + "%s_%s_avgweight.pt" % ('gcn', dataset_name))
                np.save(cache_path + "%s_%s_idx_train.npy" % (gia_name, dataset_name), dataset.idx_train)
                np.save(cache_path + "%s_%s_idx_val.npy" % (gia_name, dataset_name), dataset.idx_val)
                np.save(cache_path + "%s_%s_idx_test.npy" % (gia_name, dataset_name), dataset.idx_test)

        # obtain surrogate gnn
        if gnn_config_path is not None:
            raise NotImplementedError()
        else:
            cf_gnn = utils.get_default_gnn_config(dataset_name, 'gcn')

        surrogate_gnn = GCNModel(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'], int(dataset.labels.max() + 1),
                                dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln']).to(device)
        if load_pretraining_gnn:
            try:
                surrogate_gnn.load_state_dict(torch.load(pretraining_gnn_path+f'{dataset_name}/gcn/pretraining.pt'))
                if fine_tune:
                    _ = fit_gnn(surrogate_gnn, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                train_iters=200, patience=[20], save=False, save_path=None, save_step=np.inf)
            except:
                _ = fit_gnn(surrogate_gnn, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                            train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
                if not os.path.exists(pretraining_gnn_path+f'{dataset_name}/gcn/'):
                    os.makedirs(pretraining_gnn_path+f'{dataset_name}/gcn/')
                torch.save(surrogate_gnn.state_dict(), pretraining_gnn_path+f'{dataset_name}/gcn/pretraining.pt')
        else:
            _ = fit_gnn(surrogate_gnn, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                        train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
            if not os.path.exists(pretraining_gnn_path+f'{dataset_name}/gcn/'):
                os.makedirs(pretraining_gnn_path+f'{dataset_name}/gcn/')
            torch.save(surrogate_gnn.state_dict(), pretraining_gnn_path+f'{dataset_name}/gcn/pretraining.pt')

        # _ = fit_gnn(surrogate_gnn, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
        #             train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
            

        eva_acc_list = np.zeros((n_gnn, n_test))
        poi_acc_list = np.zeros((n_gnn, n_test))
        eva_loss_list = np.zeros((n_gnn, n_test))
        poi_loss_list = np.zeros((n_gnn, n_test))
        epochs = range(n_test)
        for epoch in epochs:
            print(f'*----* Epoch: {epoch} *----*')

            # train victim gnn
            victim_gnns = []
            for gi, gnn_name in enumerate(gnn_names):

                if args.gnn_config_path is not None:
                    raise NotImplementedError('Todo')
                else:
                    cf_gnn = utils.get_default_gnn_config(dataset_name, gnn_name)

                if gnn_name == 'gcn':
                    model = GCNModel(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'], int(dataset.labels.max()+1),
                                     dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln']).to(device)
                elif gnn_name == 'sgc':
                    model = SGCModel(dataset.features.shape[1], cf_gnn['n_layer'], int(dataset.labels.max()+1),
                                     dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln'], conv_cached=True).to(device)
                elif gnn_name == 'gat':
                    model = GATModel(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'], int(dataset.labels.max()+1),
                                     cf_gnn['n_head'], dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln']).to(device)
                elif gnn_name == 'graphsage':
                    model = GraphSAGEModel(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'],
                                           int(dataset.labels.max() + 1),
                                           dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln']).to(device)
                elif gnn_name == 'gcnguard':
                    model = GCNGuard(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'], int(dataset.labels.max()+1),
                                     dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln']).to(device)
                elif gnn_name == 'simpgcn':
                    model = SimPGCNModel(dataset.features.shape[1], cf_gnn['hidden_channels'], cf_gnn['n_layer'],
                                         int(dataset.labels.max() + 1),
                                         dropout=cf_gnn['dropout'], with_bn=cf_gnn['with_bn'], with_ln=cf_gnn['with_ln'],
                                         knn_cached=cf_gnn['knn_cached']).to(device)
                else:
                    raise NotImplementedError(gnn_name)

                if load_pretraining_gnn:
                    try:
                        model.load_state_dict(torch.load(pretraining_gnn_path+f'{dataset_name}/{gnn_name}/pretraining.pt'))
                        if fine_tune:
                            _ = fit_gnn(model, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                        train_iters=200, patience=[20], save=False, save_path=None, save_step=np.inf)
                    except:
                        _ = fit_gnn(model, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                    train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
                        if not os.path.exists(pretraining_gnn_path+f'{dataset_name}/{gnn_name}/'):
                            os.makedirs(pretraining_gnn_path+f'{dataset_name}/{gnn_name}/')
                        torch.save(model.state_dict(), pretraining_gnn_path+f'{dataset_name}/{gnn_name}/pretraining.pt')
                else:
                    _ = fit_gnn(model, pyg_data.x, pyg_data.edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
                    if not os.path.exists(pretraining_gnn_path+f'{dataset_name}/{gnn_name}/'):
                        os.makedirs(pretraining_gnn_path+f'{dataset_name}/{gnn_name}/')
                    torch.save(model.state_dict(), pretraining_gnn_path+f'{dataset_name}/{gnn_name}/pretraining.pt')
                victim_gnns.append(model)

            # attack
            if type(gia_config_path) == dict:
                cf_gia = gia_config_path[dataset_name]
            else:
                cf_gia = utils.get_default_gia_config(dataset_name, gia_name)
                cf_gia['a1'] = sp_a
                cf_gia['a2'] = 1-sp_a
                print(cf_gia)
            if not load_ptb_graphs:
                if gia_name == 'lpgia':
                    # 降低不确定性解决方案二：C&S/LP
                    surrogate_output = predict_gnn(surrogate_gnn, pyg_data.x, pyg_data.edge_index, device)
                    surrogate_output = torch.exp(surrogate_output).detach()
                    lp_model = LabelPropagation(num_layers=50, alpha=0.9,
                                                memory=False, add_self_loops=True, auto_convergence=False)
                    lp_output = lp_model(surrogate_output, pyg_data.edge_index.to(device))
                    lp_output = (lp_output / lp_output.sum(dim=1).reshape(-1, 1)).detach()  # 预测概率缓存
                    out_gcn_soft_avg = lp_output
                    if weight_gcn_avg is None:
                        weight_gcn_avg = surrogate_gnn.get_linear_weight()
                    attack_model = LPGIA(surrogate_gnn, n_top_k=10,
                                         y_score_alpha=cf_gia['a1'], bwc_score_alpha=cf_gia['a2'], bp_score_alpha=-cf_gia['a2'])
                    mod_adj, mod_features = attack_model.attack(dataset.adj, dataset.features, dataset.labels,
                                                                dataset.idx_train, dataset.idx_val, dataset.idx_test, label_access='query',
                                                                n_fake_nodes=n_fake_nodes, device=device, verbose=False,
                                                                surrogate_out_cached=out_gcn_soft_avg, weight_cached=weight_gcn_avg, 
                                                                opt_fake_feature=cf_gia['opt_fake_feature'], opt_homophily_score=cf_gia['opt_homophily_score'], opt_cluster=cf_gia['opt_cluster'],
                                                                feature_budget=cf_gia['feature_budget'], sparse_feature=cf_gia['sparse_feature'])
                else:
                    raise NotImplementedError()

                # save ptb graph
                if save_ptb_graphs:
                    # atk_name = attack_model.__class__.__name__
                    # atk_name = gia_name
                    atk_id = 0
                    tmp_path = ptb_graph_path + '%s/%s/%s/' % (dataset_name, gia_name, str(atk_id))
                    while os.path.exists(tmp_path):
                        atk_id += 1
                        tmp_path = ptb_graph_path + '%s/%s/%s/' % (dataset_name, gia_name, str(atk_id))
                    os.makedirs(tmp_path)
                    mod_adj_ts = utils.transform_sparse_mat_type(mod_adj, 'tensor')
                    torch.save(mod_adj_ts, tmp_path + "%s_%s_%s_adj.pt" % (gia_name, dataset_name, n_fake_nodes))
                    mod_feat_ts = torch.FloatTensor(mod_features)
                    torch.save(mod_feat_ts, tmp_path + "%s_%s_%s_features.pt" % (gia_name, dataset_name, n_fake_nodes))
                    np.save(tmp_path + "%s_%s_%s_idx_train" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_train)
                    np.save(tmp_path + "%s_%s_%s_idx_val" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_val)
                    np.save(tmp_path + "%s_%s_%s_idx_test" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_test)
                    del mod_adj_ts
                    del mod_feat_ts
            else:
                ptb_graph_load_path = ptb_graph_path + '%s/%s/' % (dataset_name, gia_name)
                folder_names = [folder for folder in os.listdir(ptb_graph_load_path)
                                if os.path.isdir(os.path.join(ptb_graph_load_path, folder))]
                if epoch >= len(folder_names):
                    break
                else:
                    tmp_path = ptb_graph_load_path + f'{folder_names[epoch]}/'
                    mod_adj_ts = torch.load(tmp_path + "%s_%s_%s_adj.pt" % (gia_name, dataset_name, n_fake_nodes)).coalesce()
                    mod_feat_ts = torch.load(tmp_path + "%s_%s_%s_features.pt" % (gia_name, dataset_name, n_fake_nodes))
                    mod_adj = utils.transform_sparse_mat_type(mod_adj_ts, 'csr')
                    mod_features = mod_feat_ts.numpy() if not mod_feat_ts.is_sparse else mod_feat_ts.to_dense().numpy()
                    # idx_train = np.load(tmp_path + "%s_%s_%s_idx_train" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_train)
                    # idx_val = np.load(tmp_path + "%s_%s_%s_idx_train" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_val)
                    # idx_test = np.load(tmp_path + "%s_%s_%s_idx_train" % (gia_name, dataset_name, n_fake_nodes), dataset.idx_test)
                    # pyg_data.idx_train = idx_train
                    # pyg_data.idx_test = idx_test
                    # pyg_data.idx_val = idx_val

            # evaluation
            mod_x = torch.FloatTensor(mod_features.copy()).to(device)
            mod_edge_index = utils.transform_sparse_mat_type(mod_adj, 'edge_index')

            for gi, gnn in enumerate(victim_gnns):
                # evasion
                if evasion:
                    if hasattr(gnn, 'conv_cached') and gnn.conv_cached:
                        try:
                            gnn.clear_conv_cache()
                        finally:
                            print('evasion attack need to clear conv cache')

                    output, loss, acc = test_gnn(gnn, mod_x, mod_edge_index, pyg_data.y, pyg_data.idx_test, device)
                    eva_acc_list[gi, epoch] = acc
                    eva_loss_list[gi, epoch] = loss
                # poisoning
                if poisoning:
                    if hasattr(gnn, 'conv_cached') and gnn.conv_cached:
                        try:
                            gnn.clear_conv_cache()
                        finally:
                            print('evasion attack need to clear conv cache')

                    output = fit_gnn(gnn, mod_x, mod_edge_index, pyg_data.y, pyg_data.idx_train, pyg_data.idx_val, device,
                                     train_iters=500, patience=[50], save=False, save_path=None, save_step=np.inf)
                    output, loss, acc = test_gnn(gnn, mod_x, mod_edge_index, pyg_data.y, pyg_data.idx_test, device)
                    poi_acc_list[gi, epoch] = acc
                    poi_loss_list[gi, epoch] = loss

        # save result
        for gi, gnn_name in enumerate(gnn_names):
            res_df = pd.concat([res_df,
                                pd.DataFrame({'dataset': dataset_name, 'model': gnn_name, 'range': 'avg',
                                             'eva_acc': np.mean(eva_acc_list[gi, :]), 'poi_acc': np.mean(poi_acc_list[gi, :])},
                                             index=[0])],
                               ignore_index=True)
            res_df = pd.concat([res_df,
                                pd.DataFrame({'dataset': dataset_name, 'model': gnn_name, 'range': 'max',
                                             'eva_acc': np.max(eva_acc_list[gi, :]), 'poi_acc': np.max(poi_acc_list[gi, :])},
                                             index=[0])],
                               ignore_index=True)
            res_df = pd.concat([res_df,
                                pd.DataFrame({'dataset': dataset_name, 'model': gnn_name, 'range': 'min',
                                             'eva_acc': np.min(eva_acc_list[gi, :]), 'poi_acc': np.min(poi_acc_list[gi, :])},
                                             index=[0])],
                               ignore_index=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(res_df[res_df['dataset'] == dataset_name])
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

    # report res
    res_df['eva_acc'] = (res_df['eva_acc'] * 100).round(2)
    res_df['poi_acc'] = (res_df['poi_acc'] * 100).round(2)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(res_df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    #
    if save_result:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        res_df.to_excel(result_path+'results.xlsx')


if __name__ == '__main__':
    gia_test(gia_name='lpgia', n_test=50, dataset_names=['cora', 'cora_ml', 'citeseer', 'pubmed'], gnn_names=['gcn', 'sgc', 'gat', 'gcnguard', 'simpgcn'], ptb_rate=0.05,
             save_result=True, load_cache=True)


