import os
import pandas as pd
import numpy as np
import random
import json

seed = 20211024
random.seed(seed)
np.random.seed(seed)

def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)

def readtxt(fp):
    doc = []
    file = open(fp, 'r')
    for line in file.readlines():
        line = line.strip('\n')
        doc.append(line)
    file.close()
    return doc

def save_text(fp, lists):
    f = open(fp, 'w')
    for li in lists:
        if isinstance(li, list):
            f.write(','.join(map(str, li)) + '\n')
        else:
            f.write(str(li))
            f.write('\n')
    f.close()

def pos_neg_Idx(fp):

    dti = np.loadtxt(fp)
    posIdx = dti.nonzero()
    negIdx = np.where(dti == 0)

    return posIdx, negIdx

def splits(pos_index, K):
    length = len(pos_index[1])
    ranges = list(range(length))
    random.shuffle(ranges)

    num = int(length / K)

    folds = {}
    for i in range(K):
        start = i * num
        end = (i + 1) * num
        if i == K - 1:
            fold_ls = ranges[start:]
        else:
            fold_ls = ranges[start:end]
        x = pos_index[0][fold_ls]
        y = pos_index[1][fold_ls]
        fold_i = [x.tolist(), y.tolist()]
        folds['fold_' + str(i)] = fold_i
    return folds
    #
    # return

def get_dpp_idx(alledges, pos_folds, neg_folds, save_dir):

    dpp2idx = {}
    ii = 0
    for dpp in alledges:
        dpp2idx[(dpp[0], dpp[1])] = ii
        ii = ii + 1

    len_folds = len(pos_folds)
    for i in range(len_folds):
        pos_fold_i = pos_folds['fold_' + str(i)]
        neg_fold_i = neg_folds['fold_' + str(i)]
        pos_fold_i = np.array(pos_fold_i).T
        neg_fold_i = np.array(neg_fold_i).T
        fold_i = np.concatenate([pos_fold_i, neg_fold_i], axis=0)

        dpp_idx = []
        for dpp in fold_i:
            dpp_idx.append(dpp2idx[(dpp[0], dpp[1])])
        np.savetxt(save_dir + 'fold00{}_dpp_id.txt'.format(str(i)), dpp_idx, fmt='%d')


def fold_split(pos_index, neg_index, K, save_dir):

    pos_len = len(pos_index[1])

    # positive folds split
    pos_folds = splits(pos_index, K)
    json.dump(pos_folds, open(save_dir + 'pos_folds.json', 'w'))

    # negative folds split
    num_neg_folds = [1] #, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'all']
    for i, neg_num in enumerate(num_neg_folds):
        if neg_num == 'all':
            neg_list = neg_index
        else:
            neg_samp_id = np.random.choice(np.arange(len(neg_index[0])), size=neg_num * pos_len, replace=False)
            neg_list = [neg_index[0][neg_samp_id], neg_index[1][neg_samp_id]]

        # all drug-target pairs
        posedge = np.array(list(pos_index)).T
        negedge = np.array(neg_list).T
        alledges = np.concatenate([posedge, negedge], axis=0)
        labels = np.zeros(len(alledges), int)
        labels[:len(posedge)] = 1

        fp_i = save_dir + '/neg_folds_times_{}/'.format(str(num_neg_folds[i]))
        make_dir(fp_i)
        np.savetxt(fp_i + 'DPPs.txt', alledges, fmt='%d')
        np.savetxt(fp_i + 'DPP_labels.txt', labels, fmt='%d')

        neg_folds = splits(neg_list, K)
        json.dump(neg_folds, open(save_dir + 'neg_folds_times_{}.json'.format(str(num_neg_folds[i])), 'w'))

        get_dpp_idx(alledges, pos_folds, neg_folds, fp_i)

    print('The folds split has been finished!')


if __name__ == '__main__':
    dataset = 'data'
    dti_dir = './{}/mat_data/mat_drug_protein.txt'.format(dataset)
    # fold split, 5-folds/10-folds
    # save_dir = './{}/independent_test/6_folds/'.format(dataset)  # the last fold as the test set
    save_dir = './{}/10_folds/'.format(dataset)
    make_dir(save_dir)

    # get positive pair index and negative pair index
    pos_index, neg_index = pos_neg_Idx(dti_dir)
    fold_split(pos_index, neg_index, K=10, save_dir=save_dir)
