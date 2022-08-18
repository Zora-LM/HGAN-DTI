#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import sys
import os
import math
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import torch
import torch.nn.functional as F
from kgembedUtils.kgutils import build_graph
from kgembedUtils.utils import set_seeds
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
#++++++++
from codes_kge.MAGNAKGEModel import KGEModel
from kge_losses.lossfunction import MSELoss, BCESmoothLoss
from data_preprocess import Data, fold_train_test_idx

from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', default='cuda:1', action='store_true', help='use GPU')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--nFold', type=int, default=10)
    parser.add_argument('--neg_times', type=int, default=1)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--combine_sim_network', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default='../hetero_dataset/{}/')
    parser.add_argument('-save', '--save_dir',
                        default='../results_new/{}/alpha{}_edge_drop{}_hops{}_layers{}_topk{}_head{}_loss_type{}_relation/',
                        type=str)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="adam second beta value")

    parser.add_argument('--smoothing', default=0.01, type=float, help='smoothing factor')
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay of adam")
    parser.add_argument('--regularization', default=1.0, type=float)
    parser.add_argument("--att_drop", type=float, default=0., help="attention drop out")
    parser.add_argument("--input_drop", type=float, default=0., help="input feature drop out")
    parser.add_argument("--fea_drop", type=float, default=0., help="feature drop out")
    parser.add_argument("--loss_type", type=str, default='BCE', help="MSE, BCE, BCEsmooth, MSESmooth")
    parser.add_argument("--topk_type", type=str, default='local', help="top k type, option: local")
    parser.add_argument('--max_steps', default=2, type=int)

    parser.add_argument("--alpha", type=float, default=0.05, help="random walk with restart")
    parser.add_argument("--edge_drop", type=float, default=0.1, help="graph edge drop out")
    parser.add_argument("--top_k", type=int, default=10, help="top k")
    parser.add_argument("--hops", type=int, default=2, help="hop number")
    parser.add_argument("--layers", type=int, default=6, help="number of layers")
    parser.add_argument('--num_heads', default=8, type=int)

    parser.add_argument('-cpu', '--cpu_num', default=0, type=int)
    parser.add_argument('-d', '--hidden_dim', default=128, type=int)
    parser.add_argument('-ee', '--ent_embed_dim', default=256, type=int)
    parser.add_argument('-er', '--rel_embed_dim', default=256, type=int)
    parser.add_argument('-e', '--embed_dim', default=256, type=int)
    parser.add_argument("--slope", type=float, default=0.2, help="leaky relu slope")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument('--patience', type=int, default=30, help="used for early stop")
    parser.add_argument('--feed_forward', type=int, default=1, help="0: no, 1: yes")

    parser.add_argument("--graph_on", type=int, default=1, help="Using graph")
    parser.add_argument("--trans_on", type=int, default=0, help="Using transformer")
    parser.add_argument("--mask_on", type=int, default=1, help="Using graph")
    parser.add_argument('--project_on', default=1, type=int)
    parser.add_argument('--inverse_relation', default=True, type=bool)
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--self_loop", type=int, default=1, help="self loop")

    return parser.parse_args()

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def graph_construction(args, nets, idx=None):
    with_cuda = args.cuda

    graph, num_one_dir_edges, num_relation = build_graph(args, net=nets, idx=idx)
    print('Graph information (nodes = {}, edges={})'.format(graph.number_of_nodes(), graph.number_of_edges()))

    if with_cuda:
        for key, value in graph.ndata.items():
            graph.ndata[key] = value.to(args.cuda)
        for key, value in graph.edata.items():
            graph.edata[key] = value.to(args.cuda)
    return graph, num_one_dir_edges, num_relation

def get_loss(y, logits):
    loss = None
    if args.loss_type == 'BCE':
        y_label = torch.LongTensor(y).to(logits.device)
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), y_label)
    elif args.loss_type == 'BCESmooth':
        loss_bce_smooth = BCESmoothLoss(args.smoothing)
        y_label = torch.FloatTensor(y).to(logits.device)
        # y_pred, _ = torch.max(logits, dim=1)
        loss = loss_bce_smooth(y_label, logits)
    elif args.loss_type == 'MSE':
        y_pred, _ = torch.max(torch.sigmoid(logits), dim=1)
        y_truth = torch.FloatTensor(y).to(logits.device)
        loss = F.mse_loss(y_truth, y_pred)
    elif args.loss_type == 'MSESmooth':
        loss_mse_smooth = MSELoss(args.smoothing)
        y_pred, _ = torch.max(torch.sigmoid(logits), dim=1)
        y_truth = torch.FloatTensor(y).to(logits.device)
        loss = loss_mse_smooth(y_truth, y_pred)

    return loss

def training(model, graph, optimizer, train_idx, drug_protein, type_mask):
    model.train()
    logits = model(graph=graph, type_mask=type_mask, index=train_idx)
    y = drug_protein[train_idx[0], train_idx[1]]
    loss = get_loss(y, logits)    # loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

def test(model, graph, test_idx, drug_protein, type_mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph=graph, type_mask=type_mask, index=test_idx)
        y = drug_protein[test_idx[0], test_idx[1]]
        loss = get_loss(y, logits)

        y_logits = logits[:, 1]
        y_logits = y_logits.cpu().numpy()
    auc = roc_auc_score(y, y_logits)
    aupr = average_precision_score(y, y_logits)
    return loss, auc, aupr, y, y_logits

def main(args):
    random_seed = args.seed
    set_seeds(random_seed)

    args.data_dir = args.data_dir.format(data_set)
    args.fold_path = args.data_dir + '/{}_folds/'.format(args.nFold)
    pos_folds = json.load(open(args.fold_path + 'pos_folds.json', 'r'))
    neg_folds = json.load(open(args.fold_path + 'neg_folds_times_{}.json'.format(args.neg_times, 'r')))

    net_data = Data(args)
    args.nentity = net_data.num_nodes
    graph_all, _, _ = graph_construction(args, net_data, idx=None)

    f_csv = open(args.save_dir + 'results.csv', 'a')
    f_csv.write('Fold,AUC,AUPR\n')
    f_csv.close()

    results = {'fold_' + str(i): {} for i in range(args.nFold)}
    for fold in range(args.nFold):
        print('\nThis is Fold ', fold, '...')
        if os.path.exists(args.save_dir + '/checkpoint/checkpoint_fold_{}_best.pt'.format(fold)):
            print('The training of this fold has been completed!\n')
            continue
        train_fold_idx, test_fold_idx = fold_train_test_idx(pos_folds, neg_folds, args.nFold, fold)
        graph_train, num_one_dir_edges, num_relation = graph_construction(args, net_data, idx=train_fold_idx)

        args.nrelation = num_relation
        args.nedges = num_one_dir_edges

        model = KGEModel(args)
        if args.cuda:
            model = model.to(args.cuda)

        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                         betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_steps, eta_min=1e-7)

        best_auc = 0
        best_aupr = 0
        pred = None
        counter = 0
        # Training Loop
        if os.path.exists(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold)):
            print('Load mdeol weights from /checkpoint/checkpoint_fold_{}.pt'.format(fold))
            model.load_state_dict(
                torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold), map_location=args.cuda))

        for epoch in range(args.num_epoch):
            training(model, graph_train, optimizer, train_fold_idx, net_data.drug_protein, net_data.type_mask)
            # scheduler.step()

            loss, auc, aupr, y, y_pred = test(model, graph_all, test_fold_idx, net_data.drug_protein, net_data.type_mask)
            print('Epoch {:d} | loss {:.6f} | auc {:.4f} | aupr {:.4f}'.format(epoch, loss, auc, aupr))

            torch.cuda.empty_cache()

            # early stopping
            if best_auc < auc:
                best_auc = auc
                counter = 0
            if best_aupr < aupr:
                best_aupr, pred = aupr, y_pred
                torch.save(model.state_dict(), args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold))
                counter = 0
            else:
                counter += 1

            if counter > args.patience:
                print('Early stopping!')
                break

        f_csv = open(args.save_dir + 'results.csv', 'a')
        f_csv.write(','.join(map(str, [fold, best_auc, best_aupr])) + '\n')
        f_csv.close()
        best_weights = torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold), map_location=args.cuda)
        torch.save(best_weights, args.save_dir + '/checkpoint/checkpoint_fold_{}_best.pt'.format(fold))

        results['fold_{}'.format(fold)]['pred'] = pred.tolist()
        results['fold_{}'.format(fold)]['ground_truth'] = y.tolist()
        results['fold_{}'.format(fold)]['AUC'] = best_auc.item()
        results['fold_{}'.format(fold)]['AUPR'] = best_aupr.item()

    res = pd.read_csv(args.save_dir + 'results.csv')
    try:
        auc_list = [float(res[res['Fold'] == i]['AUC'].values[0]) for i in range(args.nFold)]
        aupr_list = [float(res[res['Fold'] == i]['AUPR'].values[0]) for i in range(args.nFold)]
    except:
        auc_list = [float(res[res['Fold'] == str(i)]['AUC'].values[0]) for i in range(args.nFold)]
        aupr_list = [float(res[res['Fold'] == str(i)]['AUPR'].values[0]) for i in range(args.nFold)]

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AUPR_mean = {}, AUPR_std = {}'.format(np.mean(aupr_list), np.std(aupr_list)))

    auc_list.append(np.mean(auc_list))
    auc_list.append(np.std(auc_list))
    aupr_list.append(np.mean(aupr_list))
    aupr_list.append(np.std(aupr_list))

    json.dump(results, open(args.save_dir + '/pred_results.json', 'w'))
    np.savetxt(args.save_dir + 'AUC.txt', auc_list, fmt='%.4f')
    np.savetxt(args.save_dir + 'AUPR.txt', aupr_list, fmt='%.4f')

if __name__ == '__main__':
    args = parse_args()
    data_set = 'data_luo'
    if args.topk_type == 'local':
        args.save_dir = args.save_dir.format(data_set, args.alpha, args.edge_drop, args.hops, args.layers, args.top_k,
                                             args.num_heads, args.loss_type)
    else:
        args.save_dir = args.save_dir.format(data_set, args.alpha, args.edge_drop, args.hops, args.layers, -1,
                                             args.num_heads, args.loss_type)
    os.makedirs(args.save_dir + '/checkpoint/', exist_ok=True)

    sys.stdout = Logger(args.save_dir + 'log.txt')

    print('Save path ', args.save_dir)
    main(args)
    print('Save path ', args.save_dir)
