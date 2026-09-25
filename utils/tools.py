# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def seed_everything(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def neighbor_prototype(feature, mem_feat, idx, k, cross_domain=True):
    dis = -torch.mm(feature.detach(), mem_feat.t())  # [36, 256]x[256, 4365]=[36, 4365]
    if not cross_domain:
        for di in range(dis.size(0)):
            # set the max distance for self
            dis[di, idx[di]] = torch.max(dis)
    _, p1 = torch.sort(dis, dim=1)  # from low to high
    w = torch.zeros(feature.size(0), mem_feat.size(0)).cuda()
    # k = 5
    for wi in range(w.size(0)):
        for wj in range(k):
            w[wi][p1[wi, wj]] = 1 / k
    feat_t = w.mm(mem_feat)
    return feat_t, p1[:, 0:k]


def neighbor_select(feature, mem_feat, idx, k, cross_domain=True, gpu=True):
    torch.cuda.empty_cache()
    dis = -torch.matmul(feature, mem_feat.cuda().t())  # [36, 256]x[256, 4365]=[36, 4365]
    if not cross_domain:
        for di in range(dis.size(0)):
            # set the max distance for self
            dis[di, idx[di]] = 1
    _, p1 = torch.sort(dis if gpu else dis.cpu(), dim=1)  # from low to high
    p1 = p1[:, 0:k].cuda()
    feats = torch.index_select(mem_feat.cuda(), 0, p1.reshape(-1))
    feats = feats.reshape((feature.size(0), -1, feature.size(1)))
    return feats, p1


def maxMatch(sCenters, tCenters, s2tEdges, t2sEdges):
    s2t = [-1] * sCenters
    t2s = [-1] * tCenters
    visited = [False] * tCenters
    for s in range(sCenters):
        if s2t[s] == -1:
            visited = [False] * tCenters
            line(s, sCenters, tCenters, s2tEdges, t2sEdges, s2t, t2s, visited)
    return s2t, t2s


def line(s, sCenters, tCenters, s2tEdges, t2sEdges, s2t, t2s, visited):
    for t in range(tCenters):
        if t in s2tEdges[s] and s in t2sEdges[t] and not visited[t]:
            visited[t] = True
            if t2s[t] == -1 or line(t2s[t], sCenters, tCenters, s2tEdges, t2sEdges, s2t, t2s, visited):
                s2t[s] = t
                t2s[t] = s
                return True
    return False


def Kuhn_Munkras(sCenters, tCenters, s2tWeight):
    n = max(sCenters, tCenters)
    if n > sCenters:
        s2tWeight.extend([[-1] * n] * (tCenters - sCenters))
    ls = [0] * n
    lt = [0] * n
    for s in range(n):
        ls[s] = max(s2tWeight[s])
    match = [-1] * n
    for s in range(n):
        while True:
            searchS = [False] * n
            searchT = [False] * n
            if search_path(s, n, searchS, searchT, ls, lt, match, s2tWeight):
                break
            inc = 1000
            for i in range(n):
                if searchS[i] is True:
                    for j in range(n):
                        if not searchT[j] and ((ls[i] + lt[j] - s2tWeight[i][j]) < inc):
                            inc = ls[i] + lt[j] - s2tWeight[i][j]
            for i in range(n):
                if searchS[i]:
                    ls[i] -= inc
                if searchT[i]:
                    lt[i] += inc
    # match[i] t -> s
    inter = []
    sum = 0
    for i in range(n):
        # i is t node
        if match[i] >= sCenters:
            match[i] = -1
        elif s2tWeight[match[i]][i] == -1:
            match[i] = -1
        else:
            sum += s2tWeight[match[i]][i]
            inter.append((match[i], i))
    return inter, sum


def search_path(s, n, searchS, searchT, ls, lt, match, s2tWeight):
    searchS[s] = True
    for i in range(n):
        if not searchT[i] and ls[s] + lt[i] == s2tWeight[s][i]:
            searchT[i] = True
            if match[i] == -1 or search_path(match[i], n, searchS, searchT, ls, lt, match, s2tWeight):
                match[i] = s
                return True
    return False


