import torch
import os.path as osp
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import operator
import math
import torch.optim as optim
from utils.optimize import *
from easydict import EasyDict as edict
from utils import *
from dataset import *
import sklearn
from sklearn.cluster import KMeans


class BaseTrainer(object):
    def __init__(self, config):
        # for k-means based pseudo-label generation
        self.center_history = []  # save the history optimal num of k-means centers for the target domain
        self.common_cluster_set = None  # the clusters matching with source cluster i.e., known clusters
        self.final_n_center = None
        self.gt_freq = None
        self.num_centers = None

        self.config = config

        self.best = 0.0
        self.s_centers = None
        self.t_centers = None
        self.im_feat_s = None
        self.im_feat_t = None
        self.s_labels = None
        self.acc_best_h = 0.0
        self.h_best_acc = 0.0
        self.k_best = 0.0
        self.h_best = 0.0
        self.label_mask = None
        self.k_converge = False
        self.score_vec = None
        self.target_private_cluster2class = {}
        self.private_clusters = []
        self.test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=500,
                                       test=True, validate=True)  # test data
        self.src_loader = get_dataset(self.config, self.config.source, self.config.source_classes, batch_size=500,
                                      test=True)
        self.tgt_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=500,
                                      test=True)  # all data
        self.gt = None
        self.open_detect = None

        dim = 0
        if self.config.model != 'res50':
            dim = 100
        else:
            dim = 256
        if not self.config.bottleneck:
            dim = 2048

        self.best_prec = 0.0
        self.best_recall = 0.0

    def forward(self):
        pass

    def backward(self):
        pass

    def iter(self):
        pass

    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq == 0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq == 0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq == 0 and i_iter != 0:
                self.validate()

    def display_data(self, name, value, i_iter, display=True):
        if self.config.tensorboard:
            self.writer.add_scalar(name, value, i_iter)
        if display:
            print('{} is {:.2f}'.format(name, value))

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{}'.format(key, self.losses[key]) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor + '  ' + loss_infor)
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/' + key, self.losses[key], iter)

    def print_acc(self, acc_dict):
        str_dict = [str(k) + ': {:.2f}'.format(v) for k, v in acc_dict.items()]
        output = ' '.join(str_dict)
        print(output)

    def cos_simi(self, x1, x2):
        simi = torch.matmul(x1, x2.transpose(0, 1))
        return simi

    def gather_feats(self):
        print('gather target features')
        data_feat, data_gt, data_paths, data_probs, softmax_probs = [], [], [], [], []
        gts = []
        preds = []
        gt = {}
        names = []
        t_labels = []
        for _, batch in enumerate(self.tgt_loader):
            img, label, name, _, _ = batch
            label = label.cuda().squeeze()
            names.extend(name)
            with torch.no_grad():
                _, output, prob, softmax_prob = self.model(img.cuda())
            feature = output.squeeze()  # view(1,-1)
            N, C = feature.shape
            t_labels.extend(label.tolist())
            data_feat.extend(torch.chunk(feature, N, dim=0))
            softmax_probs.extend(torch.chunk(softmax_prob, N, dim=0))
            gts.extend(torch.chunk(label, N, dim=0))
            pred = torch.argmax(softmax_prob, dim=-1)
            preds.extend(torch.chunk(pred, N, dim=0))
        t_labels = torch.from_numpy(np.array(t_labels)).cuda()
        for k, v in zip(names, gts):
            gt[k] = v.cuda()
        feats = torch.cat(data_feat, dim=0)
        softmax_probs = torch.cat(softmax_probs, dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
        preds = torch.cat(preds, dim=0)

        return feats, t_labels, preds, softmax_probs, gt

    def validate(self, i_iter):
        self.open_validate()

    def open_validate(self):
        pass

    def get_src_centers(self):
        print('gather source features')
        self.model.eval()
        num_cls = self.config.num_classes
        if self.config.model != 'res50':
            s_centers = torch.zeros((num_cls, 100)).float().cuda()
        else:
            s_centers = torch.zeros((num_cls, 256)).float().cuda()
        if not self.config.bottleneck:
            s_centers = torch.zeros((num_cls, 2048)).float().cuda()

        counter = torch.zeros((num_cls, 1)).float().cuda()
        s_feats = []
        s_labels = []
        soft_probs = []
        for _, batch in enumerate(self.src_loader):
            acc_dict = {}
            img, label, _, _, index = batch
            label = label.cuda().squeeze()
            with torch.no_grad():
                _, neck, _, soft_prob = self.model(img.cuda())
            neck = F.normalize(neck, p=2, dim=-1)
            N, C = neck.shape
            s_labels.extend(label.tolist())
            soft_probs.extend(torch.chunk(soft_prob, N, dim=0))
            s_feats.extend(torch.chunk(neck, N, dim=0))
        soft_probs = torch.stack(soft_probs).squeeze()
        s_feats = torch.stack(s_feats).squeeze()
        s_labels = torch.from_numpy(np.array(s_labels)).cuda()
        for i in s_labels.unique():
            i_msk = s_labels == i
            index = i_msk.squeeze().nonzero(as_tuple=False)
            i_prob = soft_probs[index, i]
            i_feat = s_feats[index, :]
            i_center = torch.matmul(i_prob.t(), i_feat.squeeze()) / torch.sum(i_prob)
            i_feat = F.normalize(i_center, p=2, dim=1)
            s_centers[i, :] = i_feat
        s_feats = F.normalize(s_feats, p=2, dim=-1)
        if self.s_labels is None:
            self.s_labels = s_labels
        return s_centers, s_feats, s_labels

    def update_mem_feat(self, feature_source=None, feature_target=None, idx_s=None, idx_t=None):
        if feature_source is not None:
            self.im_feat_s[idx_s] = (1 - self.config.history_memory_update_rate) * self.im_feat_s[
                idx_s] + self.config.history_memory_update_rate * feature_source
        if feature_target is not None:
            self.im_feat_t[idx_t] = (1 - self.config.history_memory_update_rate) * self.im_feat_t[
                idx_t] + self.config.history_memory_update_rate * feature_target

    def sklearn_kmeans(self, feat, num_centers, init=None):
        if self.config.task in ['domainnet', 'visda']:
            return self.faiss_kmeans(feat, num_centers, init=init)
        if init is not None:
            kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.cpu().numpy())
        else:
            kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.cpu().numpy())
        center, t_codes = kmeans.cluster_centers_, kmeans.labels_
        score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
        return torch.from_numpy(center).cuda(), torch.from_numpy(t_codes).cuda(), score

    def faiss_kmeans(self, feat, K, init=None, niter=500):
        import faiss
        feat = feat.cpu().numpy()
        torch.cuda.empty_cache()
        d = feat.shape[1]
        kmeans = faiss.Kmeans(d, K, niter=niter, verbose=False, spherical=True, gpu=True)
        kmeans.train(feat)
        center = kmeans.centroids
        D, I = kmeans.index.search(feat, 1)
        center = torch.from_numpy(center).cuda()
        I = torch.from_numpy(I).cuda()
        D = torch.from_numpy(D).cuda()
        center = F.normalize(center, p=2, dim=-1)
        return center, I.squeeze(), D

    def re_clustering(self, i_iter, first=False):
        torch.cuda.empty_cache()
        print('re_clustering')
        self.model.eval()
        s_centers, s_feats, s_labels = self.get_src_centers()  # return centers at current step without momentum
        if self.im_feat_s is None:
            self.im_feat_s = s_feats
            self.s_centers = s_centers
        else:
            self.im_feat_s = (
                                     1 - self.config.history_memory_update_rate) * self.im_feat_s + self.config.history_memory_update_rate * s_feats
            self.s_centers = (
                                     1 - self.config.history_memory_update_rate) * self.s_centers + self.config.history_memory_update_rate * s_centers
        return self.cluster_matching(i_iter, first)

    def consensus_score(self, t_feats, t_codes, t_centers, s_feats, s_labels, s_centers, step):
        # Calculate the consensus score of cross-domain matching
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        t_centers = F.normalize(t_centers, p=2, dim=-1)
        simis = torch.matmul(s_centers, t_centers.transpose(0, 1))
        s_index = simis.argmax(dim=1)
        t_index = simis.argmax(dim=0)
        map_s2t = [(i, s_index[i].item()) for i in range(len(s_index))]
        map_t2s = [(t_index[i].item(), i) for i in range(len(t_index))]
        inter = [a for a in map_s2t if a in map_t2s]

        p_score = 0.0
        filtered_inter = []
        t_score = 0.0
        s_score = 0.0
        scores = []
        score_dict = {}
        score_vector = torch.zeros(s_centers.shape[0]).float().cuda()
        for i, j in inter:
            si_index = (s_labels == i).squeeze().nonzero(as_tuple=False)
            tj_index = (t_codes == j).squeeze().nonzero(as_tuple=False)
            si_feat = s_feats[si_index, :]
            tj_feat = t_feats[tj_index, :]

            try:
                s2TC = torch.matmul(si_feat, t_centers.transpose(0, 1))
                s2TC = s2TC.argmax(dim=-1)
            except Exception:
                print(si_index.shape)
                print(i)
                print(j)
            # calculate the closest center in target domain
            p_i2j = (s2TC == j).sum().float() / len(s2TC)
            t2SC = torch.matmul(tj_feat, s_centers.transpose(0, 1))
            t2SC = t2SC.argmax(dim=-1)
            p_j2i = (t2SC == i).sum().float() / len(t2SC)

            cu_score = (p_j2i + p_i2j) / 2
            score_dict[(i, j)] = (p_j2i, p_i2j)
            filtered_inter.append((i, j))
            t_score += p_j2i.item()
            s_score += p_i2j.item()
            p_score += cu_score.item()
            scores.append(cu_score.item())
            score_vector[i] += cu_score.item()

        score = p_score / len(filtered_inter)
        return score, score_vector, filtered_inter, scores, score_dict

    def consensus_score(self, t_feats, t_codes, t_preds, t_probs, t_centers, s_feats, s_labels, s_centers):
        # Calculate the consensus score of cross-domain matching
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        t_centers = F.normalize(t_centers, p=2, dim=-1)
        simis = torch.matmul(s_centers, t_centers.transpose(0, 1))  # source 2 target
        dis = -simis
        _, edge_s2t = torch.sort(dis, dim=-1)
        _, edge_t2s = torch.sort(dis.t(), dim=1)
        edge_s2t = edge_s2t[:, 0:5]
        edge_t2s = edge_t2s[:, 0:5]
        s2t = []
        t2s = []
        weight = simis.clone()
        for i, t_list in enumerate(edge_s2t):
            for t in t_list.cpu().numpy():
                s2t.append((i, t))
        for i, s_list in enumerate(edge_t2s):
            for s in s_list.cpu().numpy():
                t2s.append((s, i))
        inter = [a for a in s2t if a in t2s]
        for s in range(simis.size(0)):
            for t in range(simis.size(1)):
                if (s, t) not in inter:
                    weight[s][t] = -1
        inter, sum_sim = Kuhn_Munkras(simis.size(0), simis.size(1), weight.cpu().numpy().tolist())

        p_score = 0.0
        filtered_inter = []
        t_score = 0.0
        s_score = 0.0
        scores = []
        score_dict = {}
        score_vector = torch.zeros(s_centers.shape[0]).float().cuda()
        self.im_feat_t.cpu()
        for i, j in inter:  # (i: source, j: target)
            torch.cuda.empty_cache()
            tj_index = (t_codes == j).squeeze().nonzero(as_tuple=False)
            tj_feat = t_feats[tj_index, :].squeeze(1)

            _, t_neighbor_index = neighbor_select(tj_feat, self.im_feat_t, tj_index,
                                                  self.config.neighbor_k,
                                                  cross_domain=False,
                                                  gpu=False if self.config.task == 'visda' else True)
            tj_neighbor_preds = torch.index_select(t_preds, index=t_neighbor_index.reshape(-1), dim=0)
            tj_neighbor_preds = tj_neighbor_preds.reshape(tj_feat.size(0), self.config.neighbor_k)
            tj_mask = (t_preds[tj_index] == i).squeeze(1)
            tj_neighbor_mask = tj_neighbor_preds == i
            tj_prob = torch.index_select(t_probs, index=t_neighbor_index.reshape(-1), dim=0)[:, i].reshape(tj_feat.size(0), self.config.neighbor_k)
            tj_neighbor_mask = tj_neighbor_mask.sum(dim=-1) == self.config.neighbor_k

            cu_score = (tj_mask * tj_neighbor_mask).sum()
            score_dict[(i, j)] = (cu_score, cu_score)

            filtered_inter.append((i, j))
            # cluster matching results
            t_score += cu_score.item()
            s_score += cu_score.item()
            p_score += cu_score.item()
            scores.append(cu_score.item())
            score_vector[i] += cu_score.item()

        score = p_score / len(filtered_inter)
        t_score = t_score / len(filtered_inter)
        s_score = s_score / len(filtered_inter)
        min_score = np.min(scores)
        self.im_feat_t.cuda()
        return score, score_vector, filtered_inter, scores, score_dict

    def get_tgt_centers(self, step, s_centers, s_feats, s_labels, first=False):
        # Perform target clustering and then matching it with source clusters
        self.model.eval()
        t_norm_feats, t_labels, t_preds, softmax_probs, t_gt_dict = self.gather_feats()

        init_step = 0
        if self.config.pretrain:
            init_step = self.config.pretrain_steps - 1

        if self.config.setting in ['uda', 'osda', 'cda']:
            init_center = self.config.num_classes  # config.num_classes = config.cls_share + config.cls_src
        elif self.config.setting in ['pda']:
            init_center = max(self.config.interval, 2)
        else:
            init_center = self.config.num_classes

        max_center = self.config.num_classes * self.config.max_search

        gt_vector = t_labels
        id_, cnt = gt_vector.unique(return_counts=True)
        freq_dict = {}
        for i, cnt_ in zip(id_.tolist(), cnt.tolist()):
            freq_dict[i] = cnt_

        interval = int(self.config.interval)

        best_score = 0.0

        score_dict = {}
        inter_memo = {}
        search = True
        if self.config.search_stop and self.fix_k(self.center_history, self.config.fix_k):
            search = False
            self.k_converge = True
        n_center = init_center
        score_his = []
        t_codes_dic = {}
        t_centers_dic = {}
        sub_scores = {}
        score_dicts = {}
        up = 0
        pre_score = 0
        use_pre = False
        if search:
            if not first and self.num_centers is not None:
                use_pre = True
                pre_num_centers = self.config.num_centers
                t_centers, t_codes, _ = self.sklearn_kmeans(t_norm_feats, self.config.num_centers,
                                                            init=self.t_centers.cpu().numpy())
                mean_score, t_score, inter, scores, sub_dict = self.consensus_score(t_norm_feats, t_codes, t_preds,
                                                                                    softmax_probs,
                                                                                    t_centers,
                                                                                    s_feats, s_labels, s_centers)
                pre_score = mean_score
                inter_memo[-1] = inter
                sub_scores[-1] = scores
                score = mean_score
                score_dict[-1] = scores
                t_codes_dic[-1] = t_codes
                t_centers_dic[-1] = t_centers
                score_dicts[-1] = sub_dict
                best_score = score
                score_his.append(score)

            while search and n_center <= max_center:
                t_centers, t_codes, sh_score = self.sklearn_kmeans(t_norm_feats, n_center)
                mean_score, score_vector, inter, scores, sub_dict = self.consensus_score(t_norm_feats,
                                                                                         t_codes,
                                                                                         t_preds,
                                                                                         softmax_probs,
                                                                                         t_centers,
                                                                                         s_feats, s_labels, s_centers)
                score = mean_score
                inter_memo[n_center] = inter
                sub_scores[n_center] = scores
                score_dict[n_center] = scores
                t_codes_dic[n_center] = t_codes
                t_centers_dic[n_center] = t_centers
                score_dicts[n_center] = sub_dict

                if score > best_score:
                    use_pre = False
                    up += 1
                    self.final_n_center = n_center
                    best_score = score

                score_his.append(score)
                if use_pre and n_center <= pre_num_centers:
                    pass
                elif self.config.drop_stop and self.detect_continuous_drop(score_his, n=self.config.drop,
                                                                           con=self.config.drop_con):
                    if use_pre:
                        self.final_n_center = pre_num_centers
                    else:
                        self.final_n_center = n_center - interval * (self.config.drop - 1)  # do not select the last n
                    search = False
                n_center += interval

            inter = inter_memo[self.final_n_center] if not use_pre else inter_memo[-1]  # source_class -> cluster_index
            n_center = self.final_n_center if not use_pre else pre_num_centers
            t_centers = t_centers_dic[self.final_n_center] if not use_pre else t_centers_dic[-1]
            t_codes = t_codes_dic[self.final_n_center] if not use_pre else t_codes_dic[-1]
            final_sub_score = score_dict[self.final_n_center] if not use_pre else score_dict[-1]
            print('Num Centers: {}, up :{}'.format(n_center, up))
        else:
            t_centers, t_codes, _ = self.sklearn_kmeans(t_norm_feats, self.config.num_centers,
                                                        init=self.t_centers.cpu().numpy())
            st_score, t_score, inter, scores, sub_dict = self.consensus_score(t_norm_feats, t_codes, t_preds,
                                                                              softmax_probs,
                                                                              t_centers, s_feats, s_labels, s_centers)
            n_center = self.config.num_centers
            final_sub_score = scores

        self.center_history.append(n_center)
        self.config.num_centers = n_center
        self.display_data('cluster/num_centers', self.config.num_centers, step)
        self.t_centers = F.normalize(t_centers, p=2, dim=-1)

        names = list(t_gt_dict.keys())
        id_dict = {}
        for i in t_codes.unique().tolist():
            msk = (t_codes == i).squeeze()
            i_index = msk.nonzero(as_tuple=False)
            id_dict[i] = [names[a] for a in i_index]

        name2index = {}
        real_index = [i for i in range(t_norm_feats.size(0))]
        for name, index in zip(t_gt_dict.keys(), real_index):
            name2index[name] = index

        name2cluster = {}
        for i in t_codes.unique().tolist():
            msk = (t_codes == i).squeeze()
            i_index = msk.nonzero(as_tuple=False)
            for a in i_index:
                name2cluster[names[a]] = i
        self.cluster_label_all = t_codes.clone().long()
        self.cluster_label_all_one_hot = F.one_hot(self.cluster_label_all).float()

        return t_norm_feats, t_gt_dict, softmax_probs, t_codes, t_labels, freq_dict, inter, final_sub_score, name2cluster, name2index, id_dict

    def generate_target_pseudo_label(self, t_codes, t_gt_dict, cycle_pair, name2plabel, t_pseudo_label):
        t_names = list(t_gt_dict.keys())
        filtered_pair = []
        freq = {}
        filtered_cluster_label = {}
        cycle_pair_len = len(cycle_pair)
        for i in range(cycle_pair_len):
            reversed_i = cycle_pair_len - 1 - i
            pair = cycle_pair[reversed_i]
            s_index, t_index = pair
            t_mask = t_codes == t_index

            i_index = t_mask.squeeze().nonzero(as_tuple=False)
            i_names = [t_names[i[0]] for i in i_index.tolist()]
            filtered_pair.append((s_index, t_index))
            for n in i_names:
                name2plabel[n] = s_index
                filtered_cluster_label[n] = s_index
            freq[s_index] = len(i_index)
            t_pseudo_label = torch.masked_fill(t_pseudo_label, mask=t_mask, value=s_index)

            t_p_index_dict = {}
            for label in t_pseudo_label.unique().tolist():
                index = torch.where((t_pseudo_label - label) < 1e-8)[0]
                t_p_index_dict[label] = index.tolist()
        return name2plabel, filtered_pair, freq, t_pseudo_label, filtered_cluster_label

    def clus_acc(self, t_codes, t_gt, mapping, gt_freq, sub_score):
        # Print the status of matching
        for i, (src, tgt) in enumerate(mapping):
            mask = t_codes == tgt  # tgt cluster index
            i_gt = torch.masked_select(t_gt, mask)
            i_acc = ((i_gt == src).sum().float()) / len(i_gt)
            if src in gt_freq:
                gt_cnt = gt_freq[src]
            else:
                gt_cnt = 1.0
            recall = i_acc * len(i_gt) / gt_cnt
            print(
                '{:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f} Score:{:.2f}'.format(
                    src, i_acc.item(), (i_gt == src).sum().item(), len(i_gt), int(gt_cnt), i_acc, recall, sub_score[i]))

    def clus_private_acc(self, t_codes, t_gt, cluster2class, gt_freq):
        print(cluster2class, len(cluster2class.keys()))
        for cluster_label in cluster2class.keys():
            class_label = cluster2class[cluster_label]

            mask = t_codes == cluster_label
            i_gt = torch.masked_select(t_gt, mask)
            i_acc = ((i_gt == class_label).sum().float()) / len(i_gt)
            if class_label in gt_freq:
                gt_cnt = gt_freq[class_label]
            else:
                gt_cnt = 1.0
            recall = i_acc * len(i_gt) / gt_cnt

            print(
                'class:{:0>2d} {:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f}'.format(
                    class_label, cluster_label, i_acc.item(), (i_gt == class_label).sum().item(), len(i_gt),
                    int(gt_cnt),
                    i_acc, recall))

    def cluster_matching(self, step, first=False):
        # Clustering matching
        self.model.eval()

        torch.cuda.empty_cache()
        t_feats, t_gt_dict, softmax_probs, t_codes, t_gts, freq_dict, inter, sub_scores, name2cluster, name2index, id_dict = self.get_tgt_centers(
            step,
            self.s_centers,
            self.im_feat_s,
            self.s_labels,
            first=first)

        self.im_feat_s = self.im_feat_s.cuda()
        if self.im_feat_t is None:
            self.im_feat_t = t_feats
        else:
            self.im_feat_t = (
                                     1 - self.config.history_memory_update_rate) * self.im_feat_t + self.config.history_memory_update_rate * t_feats

        if self.gt_freq is None:
            self.gt_freq = {}
            for i in t_gts.unique():
                i_index = torch.where(t_gts == i)[0]
                self.gt_freq[i.item()] = i_index.size(0)

        self.name2plabel = dict(zip(t_gt_dict.keys(), [self.config.uk_index] * len(t_gt_dict.keys())))

        real_index = [i for i in range(t_feats.size(0))]
        t_pseudo_label = t_codes.clone()
        t_pseudo_label[:] = self.config.uk_index
        t_p_dict, filtered_pair, freq, t_pseudo_label, filtered_cluster_label = self.generate_target_pseudo_label(
            t_codes, t_gt_dict, inter, self.name2plabel, t_pseudo_label)
        if len(t_codes.unique().tolist()) != len(filtered_pair):
            self.open_detect = True
        self.pseudo_label_all = t_pseudo_label.long()
        self.pseudo_label_all_one_hot = F.one_hot(self.pseudo_label_all).float()
        p_t_index_k = t_pseudo_label < self.config.uk_index
        p_t_index_uk = ~p_t_index_k
        p_t_index_k = p_t_index_k.nonzero()
        p_t_index_uk = p_t_index_uk.nonzero()
        if self.history_binary is None:
            self.history_binary = torch.zeros(t_feats.size(0), 2).float().cuda()
        self.history_binary[p_t_index_k, 0] = self.history_binary[p_t_index_k, 0] + 1
        self.history_binary[p_t_index_uk, 1] = self.history_binary[p_t_index_uk, 1] + 1

        print('common clusters filtered(source, cluster index):\n {} {}'.format(filtered_pair, len(filtered_pair)))

        self.clus_acc(t_codes.squeeze(), t_gts, inter, self.gt_freq, sub_scores)

        correct = 0.0
        label_set = [i[0] for i in filtered_pair]  # common classes
        self.cluster_mapping = {i[1]: i[0] for i in filtered_pair}  # cluster_label: class_label
        self.common_cluster_set = [i[1] for i in filtered_pair]
        private_cluser = []

        for cluster_label in t_codes.unique():
            if cluster_label not in self.common_cluster_set:
                private_cluser.append(cluster_label)
                self.cluster_mapping[cluster_label.item()] = self.config.uk_index

        return self.name2plabel, name2index, label_set, name2cluster

    def fix_k(self, scores, n=3):
        # Stopping critetion: stop searching if K holds a certain value for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0
        for i in scores:
            if i == scores[-n]:
                flag += 1
        if flag == n:
            return True
        else:
            return False

    def detect_continuous_drop(self, scores, n=3, con=False):
        # Stopping Criterion: stop searching in a round if the score drops continuously for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0
        if con:
            for i in range(1, n):
                if scores[-i] <= scores[-(i + 1)]:
                    flag += 1
        else:
            flag = 0.0
            for i in scores:
                if i <= scores[-n]:
                    flag += 1
        if flag >= n - 1:
            return True
        else:
            return False
