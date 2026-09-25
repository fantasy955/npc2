import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))


def con_s(feature, label_query,
          source_memory, source_label, s_centers,
          t=0.05):
    source_all_sim = torch.matmul(feature, source_memory.t())  # (batch_size, memory_len)
    self_center_source = torch.index_select(s_centers, index=label_query, dim=0)

    loss = Variable(torch.tensor(0.).cuda())
    for i, label in enumerate(label_query):
        tp_index = torch.where(torch.abs(source_label - label) < 1e-8)[0]
        tn_idnex = torch.where(torch.abs(source_label - label) > 1e-8)[0]

        tp_sim = source_all_sim[i][tp_index]
        tn_sim = source_all_sim[i][tn_idnex]
        score = torch.exp(
            torch.cat((torch.mean(tp_sim).unsqueeze(0), tn_sim)) / t
        )
        i_loss_0 = -(torch.sum(tp_sim / t) -
                     tp_sim.size(0) * torch.log(score.sum())) / tp_sim.size(0)
        i_loss = i_loss_0

        loss += i_loss

    loss = loss / feature.size(0)
    return loss


def npc_k(feature, label_query, weight_t, weight_s,
          p_label_target, memory_target,
          label_source, memory_source,
          prototype_source, prototype_target,
          t=0.07):
    if feature.size(0) < 1e-8:
        return Variable(torch.tensor(0.).cuda())
    target_sim = torch.matmul(feature, memory_target.t())  # (batch_size, memory_len)
    target_proto_sim = torch.mul(feature, prototype_target).sum(dim=-1)

    source_sim = torch.matmul(feature, memory_source.t())
    source_proto_sim = torch.mul(feature, prototype_source).sum(dim=-1)

    loss_neighbor = Variable(torch.tensor(0.).cuda())
    loss_neighbor_t = Variable(torch.tensor(0.).cuda())
    loss_neighbor_s = Variable(torch.tensor(0.).cuda())
    for i, label in enumerate(label_query):
        # target alignment
        pn_sample_index = torch.where(torch.abs(p_label_target - label) > 1e-8)[0]
        pn_sim = target_sim[i][pn_sample_index]
        exp_all_sim = torch.exp(torch.cat((target_proto_sim[i].unsqueeze(0), pn_sim)) / t)
        i_loss_0 = -torch.log(1e-12 + torch.exp(target_proto_sim[i] / t) / torch.sum(exp_all_sim))

        # target - source alignment
        pn_sample_index = torch.where(torch.abs(label_source - label) > 1e-8)[0]
        pn_sim = source_sim[i][pn_sample_index]
        exp_all_sim = torch.exp(torch.cat((source_proto_sim[i].unsqueeze(0), pn_sim)) / t)
        i_loss_1 = -torch.log(1e-12 + torch.exp(source_proto_sim[i] / t) / torch.sum(exp_all_sim))
        # target - source alignment
        loss_neighbor_t = loss_neighbor_t + i_loss_0 * weight_t[i]
        loss_neighbor_s = loss_neighbor_s + i_loss_1 * weight_s[i]
    loss_neighbor = (loss_neighbor_t / weight_t.sum() + loss_neighbor_s / weight_s.sum()) / 2
    # loss_neighbor = loss_neighbor_t / weight_t.sum()

    return loss_neighbor


def npc_uk(feature, cluster_query, weight,
           memory_target, memory_source,
           cluster_label_all,
           prototype_target,
           t=0.07):
    if feature.size(0) < 1e-8:
        return Variable(torch.tensor(0.).cuda())
    sim_target = torch.matmul(feature, memory_target.t())  # (batch_size, memory_target_len)
    sim_source = torch.matmul(feature, memory_source.t())

    target_proto_sim = torch.mul(feature, prototype_target).sum(dim=-1)

    loss_neighbor = Variable(torch.tensor(0.).cuda())
    for i, label in enumerate(cluster_query):
        pn_index_target = torch.where(torch.abs(cluster_label_all - label) > 1e-8)[0]
        exp_all_sim = torch.exp(
            torch.cat((target_proto_sim[i].unsqueeze(0),
                       sim_target[i][pn_index_target],
                       # sim_target[i],
                       sim_source[i])) / t
        )
        i_loss = -torch.log(torch.exp(target_proto_sim[i] / t) / torch.sum(exp_all_sim))

        loss_neighbor = loss_neighbor + i_loss * weight[i]
    loss_neighbor = loss_neighbor / weight.sum()

    return loss_neighbor
