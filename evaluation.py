import os
import numpy as np
import math
import collections
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, average_precision_score

import torch
import torch.backends.cudnn as cudnn


def read_img_list(img_list_path):
    img_list = []
    labels = []
    with open(img_list_path, 'r') as file:
        for line in file.readlines():
            tmp = line.strip().split(' ')
            img_list.append(tmp[0])
            labels.append(int(tmp[1]))
    return img_list, labels


def read_pair_list(pair_list_path):
    pair_list = []
    with open(pair_list_path, 'r') as file:
        for line in file.readlines():
            tmp = line.strip().split(' ')
            pair_list.append((int(tmp[0]), int(tmp[1])))
    return pair_list


def load_feat(root_feat, img_list):
    all_feat = []
    for feat_name in tqdm(img_list):
        feat_path = os.path.join(root_feat, feat_name)
        feat_path = os.path.splitext(feat_path)[0] + '.feat'
        feat = np.fromfile(feat_path, dtype=np.float32)
        all_feat.append(feat)
    all_feat = np.vstack(all_feat)
    return all_feat


def get_match_label(label1, label2):
    if label1 == label2:
        return 1
    else:
        return 0


def get_11_scores_and_label(all_feat, pair_list, labels):
    score_matrix = cosine_similarity(all_feat, all_feat)
    score_list = []
    label_list = []
    for idx1, idx2 in pair_list:
        score = score_matrix[idx1, idx2]
        label = get_match_label(labels[idx1], labels[idx2])
        score_list.append(score)
        label_list.append(label)
    score_list = np.array(score_list)
    label_list = np.array(label_list)
    return score_list, label_list


def get_1n_scores_and_label(gallery_feat, probe_feat, gallery_label, probe_label, g_p_issame=False):
    score_matrix = cosine_similarity(gallery_feat, probe_feat)

    num_g = len(gallery_label)
    num_p = len(probe_label)

    gallery_label = np.array(gallery_label).reshape((num_g, 1))
    probe_label = np.array(probe_label).reshape((num_p, 1))

    label_matrix = get_label_matrix(gallery_label, probe_label)
    label_matrix = label_matrix.astype(np.int32)

    if g_p_issame:
        np.fill_diagonal(score_matrix, 0)
        np.fill_diagonal(label_matrix, 0)

    return score_matrix, label_matrix


def get_label_matrix(gallery_label, probe_label):
    num_g = gallery_label.shape[0]
    num_p = probe_label.shape[0]
    label_matrix = np.zeros((num_g, num_p))

    g_l = np.broadcast_to(gallery_label, (num_g, num_p))
    p_l = np.broadcast_to(probe_label.T, (num_g, num_p))
    label_matrix[np.where(g_l == p_l)] = 1
    return label_matrix


def get_tpr_at_fpr(tpr, fpr, thr):
    idx = np.argwhere(fpr > thr)
    return tpr[idx[0]][0]


def get_eer(tpr, fpr):
    for i, fpr_point in enumerate(fpr):
        if (tpr[i] >= 1 - fpr_point):
            idx = i
            break
    if (tpr[idx] == tpr[idx + 1]):
        return 1 - tpr[idx]
    else:
        return fpr[idx]


def get_rank_and_hit(score_matrix, label_matrix, ranks, g_p_issame=False):
    score_matrix = torch.from_numpy(score_matrix)
    label_matrix = torch.from_numpy(label_matrix)
    _, pred = torch.topk(score_matrix, max(ranks), 0, True, True)
    label_max = torch.gather(label_matrix, 0, pred)

    hit_list = list()

    total_num_of_hits = torch.sum(torch.sum(label_matrix, 0) > 0)

    if g_p_issame == True:
        total_num_of_hits = total_num_of_hits - 1

    for r in ranks:
        l = label_max[:r, :]
        l = torch.sum(l, dim=0) > 0
        hit = torch.sum(l)
        # hit_list.append(hit*1.0/total_num_of_hits.float())
        hit_list.append((hit * 1.0).type(torch.FloatTensor) / total_num_of_hits.float())
    return hit_list


def evaluation_pairs(root_feat, img_list_path, pair_list_path, thresholds=[1e-2, 1e-3, 1e-4, 1e-5, 0]):
    # load data
    img_list, labels = read_img_list(img_list_path)

    pair_list = read_pair_list(pair_list_path)
    all_feat = load_feat(root_feat, img_list)

    # compute score
    score_list, label_list = get_11_scores_and_label(all_feat, pair_list, labels)

    # evaluation
    ap = average_precision_score(label_list, score_list)
    print('AP:\t\t{}'.format(ap))

    fpr, tpr, _ = roc_curve(label_list, score_list, pos_label=1)

    eer = get_eer(tpr, fpr)
    print('EER:\t\t{}'.format(1 - eer))

    for thr in thresholds:
        prec = get_tpr_at_fpr(tpr, fpr, thr)
        print('TPR@FAR={}:\t{}'.format(thr, prec))


def evaluation_1n(root_feat, probe_list_path, gallery_list_path, g_p_issame=False, ranks=[1, 3, 5, 10, 20]):
    # load data
    probe_list, probe_label = read_img_list(probe_list_path)
    gallery_list, gallery_label = read_img_list(gallery_list_path)

    probe_feat = load_feat(root_feat, probe_list)
    gallery_feat = load_feat(root_feat, gallery_list)

    # compute features for cmc
    score_matrix, label_matrix = get_1n_scores_and_label(gallery_feat, probe_feat, gallery_label, probe_label,
                                                         g_p_issame=g_p_issame)
    # cmc
    hit_lst = get_rank_and_hit(score_matrix, label_matrix, ranks, g_p_issame=g_p_issame)

    for i, hit in enumerate(hit_lst):
        print('Rank-{}:\t{}'.format(ranks[i], hit))


def evaluation_11(root_feat, probe_list_path, gallery_list_path, g_p_issame=False,
                  thresholds=[1e-2, 1e-3, 1e-4, 1e-5, 0]):
    # load data
    probe_list, probe_label = read_img_list(probe_list_path)
    gallery_list, gallery_label = read_img_list(gallery_list_path)

    probe_feat = load_feat(root_feat, probe_list)
    gallery_feat = load_feat(root_feat, gallery_list)

    # compute features for cmc
    score_matrix, label_matrix = get_1n_scores_and_label(gallery_feat, probe_feat, gallery_label, probe_label,
                                                         g_p_issame=g_p_issame)

    if g_p_issame:
        # np.fill_diagonal(score_matrix, 1)
        np.fill_diagonal(label_matrix, -1)
        idx = np.where(label_matrix != -1)
        score_matrix = score_matrix[idx]
        label_matrix = label_matrix[idx]

    score_matrix = score_matrix.reshape((-1,))
    label_matrix = label_matrix.reshape((-1,))

    ap = average_precision_score(label_matrix, score_matrix)
    print('AP:\t\t{}'.format(ap))

    fpr, tpr, _ = roc_curve(label_matrix, score_matrix, pos_label=1)

    eer = get_eer(tpr, fpr)
    print('EER:\t\t{}'.format(1 - eer))

    for thr in thresholds:
        prec = get_tpr_at_fpr(tpr, fpr, thr)
        print('TPR@FAR={}:\t{}'.format(thr, prec))


### compute ROC for billion-level pair testing.
def collect_labels(label_list):
    ordered_label = collections.OrderedDict()
    current_label = label_list[0]
    _label_start = 0
    for i, label in enumerate(label_list):
        if label == current_label:
            continue
        else:
            ordered_label[current_label] = [_label_start, i - 1]
            _label_start = i
            current_label = label
    ordered_label[current_label] = [_label_start, len(label_list) - 1]
    return ordered_label


def get_total_pos_neg_num(gallery_label, probe_label, g_p_issame=False):
    num_g = len(gallery_label)
    num_p = len(probe_label)

    ordered_probe_label = collect_labels(probe_label)
    ordered_gallery_label = collect_labels(gallery_label)

    total_pos = 0
    total_neg = 0

    for _lg, range_g in ordered_gallery_label.items():
        if ordered_probe_label.has_key(_lg):
            _range_p = ordered_probe_label[_lg]
            _num_pos = (_range_p[1] - _range_p[0] + 1) * (range_g[1] - range_g[0] + 1)
            _num_neg = (range_g[1] - range_g[0] + 1) * num_p - _num_pos
        else:
            _num_pos = 0
            _num_neg = num_p * (range_g[1] - range_g[0] + 1)

        total_pos += _num_pos
        total_neg += _num_neg
    return total_pos, total_neg, ordered_probe_label, ordered_gallery_label


def compute_cosine_similarity(feat1, feat2, eps=1e-8):
    norm1 = torch.norm(feat1, 2, 1)
    norm2 = torch.norm(feat2, 2, 1)

    feat1_norm = feat1 / (norm1.view(feat1.size(0), 1) + eps)
    feat2_norm = feat2 / (norm2.view(feat2.size(0), 1) + eps)

    return torch.mm(feat1_norm, feat2_norm.t())


def evaluation_11_bigdata(root_feat, probe_list_path, gallery_list_path, g_p_issame=False,
                          thresholds=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]):
    cudnn.benchmark = True

    # load data
    probe_list, probe_label = read_img_list(probe_list_path)
    gallery_list, gallery_label = read_img_list(gallery_list_path)

    print('load probe features')
    probe_feat = load_feat(root_feat, probe_list)

    print('load gallery features')
    gallery_feat = load_feat(root_feat, gallery_list)

    probe_feat = torch.from_numpy(probe_feat).cuda()
    gallery_feat = torch.from_numpy(gallery_feat).cuda()

    total_pos, total_neg, ordered_probe_label, ordered_gallery_label = get_total_pos_neg_num(gallery_label, probe_label,
                                                                                             g_p_issame)
    print('Total pos, total neg: ', total_pos, total_neg)

    min_fp_point = min(thresholds)
    max_fp_point = max(thresholds)

    if total_neg < 1 / min_fp_point:
        raise ValueError("Not have enough samples to compute TPR@{}".format(min_fp_point))

    print('Scan all the neg scores to get the thresholds')
    neg_num_to_keep = int(math.ceil(total_neg * max_fp_point))
    top_neg_scores = torch.zeros(neg_num_to_keep).cuda()
    _check_neg_num = 0

    for _lg, _range_g in tqdm(ordered_gallery_label.items()):
        p_features = probe_feat
        g_features = gallery_feat[_range_g[0]:_range_g[1] + 1, :]

        _scores = compute_cosine_similarity(p_features, g_features)

        if ordered_probe_label.has_key(_lg):
            _range_p = ordered_probe_label[_lg]
            _num_pos = (_range_p[1] - _range_p[0] + 1) * (_range_g[1] - _range_g[0] + 1)
            _scores[_range_p[0]:_range_p[1] + 1, :] = -1.0
        else:
            _num_pos = 0

        _check_neg_num += (_scores.numel() - _num_pos)

        _scores = _scores.view(-1)

        # save top negative scores
        if _scores.numel() > neg_num_to_keep:
            _top_score, _ = torch.topk(_scores, neg_num_to_keep)
        else:
            _top_score = _scores
        top_neg_scores = torch.cat([top_neg_scores, _top_score])
        top_neg_scores, _ = torch.topk(top_neg_scores, neg_num_to_keep)
        top_neg_scores = top_neg_scores.contiguous()

    print(top_neg_scores.size())
    assert _check_neg_num == total_neg

    print('Scan all the positive scores to compute TPR')
    top_neg_scores, _ = torch.sort(top_neg_scores, descending=True)
    tp_point = list()
    for _fp in thresholds:
        thr = top_neg_scores[int(neg_num_to_keep * (_fp / max_fp_point)) - 1]
        tp_point.append(thr)

    # print('Test threshold is ', tp_point)

    _check_pos_num = 0
    tp_hit_list = [0, ] * len(thresholds)

    for _lg, _range_g in tqdm(ordered_gallery_label.items()):
        p_features = probe_feat
        g_features = gallery_feat[_range_g[0]:_range_g[1] + 1, :]

        _scores = compute_cosine_similarity(p_features, g_features)

        if ordered_probe_label.has_key(_lg):
            _range_p = ordered_probe_label[_lg]
            _pos_scores = _scores[_range_p[0]:_range_p[1] + 1, :]
            _check_pos_num += _pos_scores.numel()
        else:
            continue

        for ii in range(len(tp_hit_list)):
            tp_hit_list[ii] += (_pos_scores > tp_point[ii]).nonzero().size(0)

    assert _check_pos_num == total_pos
    tp_hit_list = [1.0 * x / total_pos for x in tp_hit_list]
    return tp_hit_list


if __name__ == '__main__':

    root_feat = './feat'
    probe_list_path = '/data1/chaoyou.fu/HFR_Datasets/CASIA_NIR_VIS_align3/list_file/NIR_VIS_probe1.txt'
    gallery_list_path = '/data1/chaoyou.fu/HFR_Datasets/CASIA_NIR_VIS_align3/list_file/NIR_VIS_gallery1.txt'

    evaluation_1n(root_feat, probe_list_path, gallery_list_path)
    evaluation_11(root_feat, probe_list_path, gallery_list_path)
