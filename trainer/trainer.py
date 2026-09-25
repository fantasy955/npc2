import os
from torch.autograd import Variable
from dataset import *
from model import *
from trainer.base_trainer import *
from utils.loss import *
from utils.tools import *


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.pseudo_label_all = None  # target pseudo label for all samples
        self.cluster_mapping = None  # associate the class label with the cluster label (output by k-means i.e., map^{t->s})
        self.name2plabel = None
        self.im_feat_t = None
        self.im_feat_s = None

    def iter(self, i_iter):
        self.losses = edict({})
        self.optimizer.zero_grad()

        inverseDecaySheduler(self.optimizer, i_iter, self.config.lr, gamma=self.config.gamma, power=self.config.power,
                             num_steps=self.config.num_steps)
        lam = i_iter / self.config.num_steps

        s_batch = next(self.s_loader)
        _, s_batch = s_batch
        s_img, s_label, _, _, s_index = s_batch
        s_img_ = s_img.cuda()
        s_label = s_label.cuda()
        _, s_neck, s_prob, _ = self.model(s_img_)
        s_loss = Variable(torch.tensor(0.).cuda())
        cet_loss = F.cross_entropy(s_prob, s_label.squeeze())

        self.losses.cet_loss = cet_loss.item()
        s_loss = s_loss + cet_loss
        s_loss.backward()

        if self.pretrain and i_iter + 1 <= self.config.pretrain_steps:
            self.optimizer.step()
            return 0
        del s_prob, s_neck, _
        torch.cuda.empty_cache()

        self.im_feat_t = self.im_feat_t.cpu()
        _, s_neck, s_prob, s_softmax_prob = self.model(s_img_)
        s_neck_norm = F.normalize(s_neck, p=2, dim=-1)
        self.update_mem_feat(feature_source=s_neck_norm.detach(), idx_s=s_index)
        source_con_loss = con_s(s_neck_norm, s_label,
                                self.im_feat_s, self.s_labels, self.s_centers,
                                t=self.config.con_gamma)
        self.losses.source_con_loss = source_con_loss.item()
        s_loss = self.config.lam_con_s * source_con_loss
        s_loss.backward()
        self.im_feat_t = self.im_feat_t.cuda()

        t_batch = next(self.t_loader)
        _, t_batch = t_batch
        t_img, cluster_label, _, _, t_index = t_batch
        n, k, c, h, w = t_img.shape
        t_img = t_img.view(-1, c, h, w).cuda()
        cluster_label = cluster_label.view(-1).cuda()
        t_index = t_index.view(-1).cuda()
        p_label = self.pseudo_label_all[t_index]

        _, t_neck, t_prob, t_softmax_prob = self.model(t_img)
        t_neck_norm = F.normalize(t_neck, p=2, dim=-1)
        t_preds = torch.argmax(t_softmax_prob, dim=-1)

        self.update_mem_feat(feature_target=t_neck_norm.detach(), idx_t=t_index)
        self.t_preds[t_index] = t_preds

        target_loss = Variable(torch.tensor(0.).cuda())
        # target contrastive learning for k and uk
        uk_sample_index = torch.where(p_label >= self.config.uk_index)[0]
        k_sample_index = torch.where(p_label < self.config.uk_index)[0]

        _, t_neighbor_index_c = neighbor_select(t_neck_norm, self.im_feat_t, t_index,
                                                self.config.neighbor_k * 10,
                                                cross_domain=False)  # similarity from high to low
        t_neighbor_plabel_c = torch.index_select(self.cluster_label_all, dim=0,
                                                 index=t_neighbor_index_c.reshape(-1))
        t_neighbor_plabel_c = t_neighbor_plabel_c.reshape((t_neighbor_index_c.size(0), -1))
        t_same_count = (t_neighbor_plabel_c == cluster_label.unsqueeze(1).repeat((1, self.config.neighbor_k * 10))).sum(
            dim=-1)
        t_same_rate = t_same_count / (self.config.neighbor_k * 10)
        t_same_rate_k = t_same_rate[k_sample_index]
        t_same_rate_uk = t_same_rate[uk_sample_index]
        weight_k = t_same_rate_k
        weight_uk = t_same_rate_uk

        prototype_target, t_neighbor_index = neighbor_prototype(t_neck_norm, self.im_feat_t, t_index,
                                                                self.config.neighbor_k,
                                                                cross_domain=False)
        prototype_source, s_neighbor_index = neighbor_prototype(t_neck_norm, self.im_feat_s, s_index,
                                                                self.config.neighbor_k,
                                                                cross_domain=True)
        if weight_k.sum() > 0:
            k_loss = npc_k(t_neck_norm[k_sample_index], p_label[k_sample_index],
                           weight_k, weight_k,
                           self.pseudo_label_all, self.im_feat_t,
                           self.s_labels, self.im_feat_s,
                           prototype_source[k_sample_index],
                           prototype_target[k_sample_index],
                           t=self.config.con_gamma)
            target_loss = target_loss + k_loss * self.config.lam_con_t
            self.losses.k_neighbor_loss = k_loss.item()
        if weight_uk.sum() > 0:
            uk_loss = npc_uk(t_neck_norm[uk_sample_index], cluster_label[uk_sample_index],
                             weight_uk,
                             self.im_feat_t, self.im_feat_s,
                             self.cluster_label_all,
                             prototype_target[uk_sample_index],
                             t=self.config.con_gamma)
            target_loss = target_loss + uk_loss * self.config.lam_con_t
            self.losses.uk_neirhbor_loss = uk_loss.item()
        # end target contrastive learning

        prob_k = t_softmax_prob[k_sample_index]
        prob_uk = t_softmax_prob[uk_sample_index]
        min_loss = self.contrastive_loss4min_entropy(prob_k)
        target_loss = target_loss + min_loss * self.config.lam_con_t
        self.losses.min_loss = min_loss.item()
        max_loss = -self.contrastive_loss4min_entropy(prob_uk)
        target_loss = target_loss + max_loss * self.config.lam_con_t
        self.losses.max_loss = max_loss.item()

        target_loss.backward()
        self.optimizer.step()

    def optimize(self):
        base_iter = 0
        for i_iter in tqdm(range(self.config.num_steps + 1)):
            self.model = self.model.train()
            # Pseudo-label generation after pre-train on the source domain or
            # after certain optimization steps, conduct pseudo-label generation
            if (not self.pretrain and i_iter == 0) or (self.pretrain and i_iter == self.config.pretrain_steps) or \
                    (i_iter > 0 and i_iter % self.config.stage_size == 0):
                # pseudo-label generation
                name2plabel, name2index, class_set, name2cluster = self.re_clustering(i_iter, first=True)
                # generate data loaders of sufficient length
                self.s_loader, _ = init_pair_dataset(self.config, length=self.config.stage_size,
                                                     plabel_dict=None, binary_label=None)
                # generate data loader for target domain with pseudo-label
                self.t_loader = init_target_dataset(self.config, length=self.config.stage_size,
                                                    plabel_dict=name2cluster,
                                                    name2index=name2index,
                                                    tgt_class_set=[i for i in range(self.config.num_centers)])
                self.validate(i_iter)

            losses = self.iter(i_iter)

            if i_iter % self.config.print_freq == 0:
                self.print_loss(i_iter)

        # at the end of DA, perform validation
        self.validate(i_iter)

    def train(self):
        self.model = init_model(self.config)
        self.nll = nn.NLLLoss()
        # self.tSNE_visual_uk(0)

        self.unknown = []
        self.model = self.model.train()
        self.center_history = []
        self.optimizer = optim.SGD(self.model.optim_parameters(
            self.config.lr) if not self.config.multi_gpu else self.model.module.optim_parameters(self.config.lr),
                                   lr=self.config.lr,
                                   momentum=self.config.momentum, weight_decay=self.config.weight_decay, nesterov=True)

        self.feat_dim = 256
        self.pretrain = self.config.pretrain

        if not self.pretrain:
            pass
        else:
            data_set_length = self.config.pretrain_steps
            self.s_loader, self.t_loader = init_pair_dataset(self.config, length=data_set_length,
                                                             plabel_dict=None, binary_label=None)

        self.optimize()

    # validation for osda and opda
    def open_validate(self, i_iter):
        print('open validate')
        self.model.train(False)
        uk_detect = False

        memo_correct = 0
        cls_correct = 0
        size = 0
        unk_class = self.config.cls_src + self.config.cls_share

        class_list = [i for i in range(self.config.cls_share)]
        class_list.append(unk_class)
        per_class_num = np.zeros((self.config.cls_share + 1))
        memo_per_class_correct = np.zeros((self.config.cls_share + 1)).astype(np.float32)
        cls_per_class_correct = np.zeros((self.config.cls_share + 1)).astype(np.float32)
        memo_detected_class_num = np.zeros((self.config.cls_share + 1))
        cls_detected_class_num = np.zeros((self.config.cls_share + 1))
        all_memo_pred = []
        all_cls_pred = []
        all_gt = []

        gts = []
        preds = []
        gt = {}
        t_labels = []
        softmax_probs = []
        names = []
        for _, batch in enumerate(self.test_loader):
            acc_dict = {}
            img_t, label_t, path_t = batch[0], batch[1], batch[2]
            names.extend(path_t)
            label = label_t.clone()
            p_binary_label = label_t.clone()
            gt_label = label_t.clone()
            idx_k = torch.where(label < self.config.uk_index)[0]
            idx_uk = torch.where(label >= self.config.uk_index)[0]
            label[idx_k] = 0
            label[idx_uk] = 1
            gt_label[idx_uk] = self.config.uk_index
            img_t, label_t = img_t.cuda(), label_t.cuda()
            label_t = torch.where(label_t >= unk_class, torch.Tensor([unk_class]).cuda(), label_t.float())
            with torch.no_grad():
                _, feat, prob, prob_softmax = self.model(img_t)
            feat = F.normalize(feat, p=2, dim=-1)
            N, C = feat.shape

            gts.extend(torch.chunk(label_t.cuda().squeeze(), N, dim=0))
            t_labels.extend(torch.chunk(label_t, N, dim=0))
            pred = torch.argmax(prob_softmax, dim=-1)
            preds.extend(torch.chunk(pred, N, dim=0))
            softmax_probs.extend(torch.chunk(prob_softmax, N, dim=0))

            simi2cluster = self.cos_simi(feat, self.t_centers)
            clus_index = simi2cluster.argmax(dim=-1)
            cls_pred = prob_softmax.argmax(-1)
            memo_pred = torch.zeros_like(clus_index)
            for cl in clus_index.unique():
                i_index = torch.where(clus_index == cl)[0]
                if cl in self.common_cluster_set:
                    memo_pred[i_index] = self.cluster_mapping[cl.item()]
                else:
                    memo_pred[i_index] = self.config.uk_index
                    uk_detect = True
            known_index = torch.where(memo_pred != self.config.uk_index)[0]
            label = label_t.clone()
            p_idx_k = torch.where(memo_pred < self.config.uk_index)[0]
            p_idx_uk = torch.where(memo_pred >= self.config.uk_index)[0]
            p_binary_label[p_idx_k] = 0
            p_binary_label[p_idx_uk] = 1

            cls_pred = torch.argmax(prob_softmax, dim=-1)
            cls_pred = torch.where(memo_pred >= self.config.uk_index, memo_pred, cls_pred)
            k = label_t.data.size()[0]
            memo_pred = memo_pred.cpu().numpy()
            cls_pred = cls_pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_memo_pred += list(memo_pred)
            all_cls_pred += list(cls_pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                memo_correct_ind = np.where(memo_pred[t_ind[0]] == t)
                cls_correct_ind = np.where(cls_pred[t_ind[0]] == t)
                memo_per_class_correct[i] += float(len(memo_correct_ind[0]))
                cls_per_class_correct[i] += float(len(cls_correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
                memo_detected_class_num[i] += float(len(np.where(memo_pred == t)[0]))
                cls_detected_class_num[i] += float(len(np.where(cls_pred == t)[0]))
                memo_correct += float(len(memo_correct_ind[0]))
                cls_correct += float(len(cls_correct_ind[0]))
            size += k
        memo_per_class_acc = memo_per_class_correct / per_class_num
        cls_per_class_acc = cls_per_class_correct / per_class_num
        memo_known_acc = memo_per_class_correct[:-1].sum() / per_class_num[:-1].sum()
        cls_known_acc = cls_per_class_correct[:-1].sum() / per_class_num[:-1].sum()
        std_memo_h_score = float(
            2 * memo_known_acc * memo_per_class_acc[-1] / (memo_known_acc + memo_per_class_acc[-1]))
        std_cls_h_score = float(2 * cls_known_acc * cls_per_class_acc[-1] / (cls_known_acc + cls_per_class_acc[-1]))

        output = ['memo-val', i_iter, list(memo_per_class_acc),
                  'per class mean acc %s' % float(memo_per_class_acc.mean()),
                  float(memo_correct / size), 'H score is %s' % std_memo_h_score,
                  'AVG Known ACC is %s' % float(memo_per_class_acc[:-1].mean()),
                  'unKnown ACC is %s' % float(memo_per_class_acc[-1])]
        print(output)
        output = ['cls-val', i_iter, list(cls_per_class_acc),
                  'per class mean acc %s' % float(cls_per_class_acc.mean()),
                  float(cls_correct / size), 'H score is %s' % std_cls_h_score,
                  'AVG Known ACC is %s' % float(cls_per_class_acc[:-1].mean()),
                  'unKnown ACC is %s' % float(cls_per_class_acc[-1])]
        print(output)

        t_labels = torch.cat(t_labels, dim=0).cuda()
        preds = torch.cat(preds, dim=0)
        softmax_probs = torch.cat(softmax_probs, dim=0)
        for k, v in zip(names, gts):
            gt[k] = v.cuda()

        return t_labels, preds, softmax_probs, gt

    def emm_loss(self, prob, w=None):
        if prob.size(0) < 1e-8:
            return Variable(torch.tensor(0.).cuda())
        if w is None:
            min_entropy = torch.mean(-torch.sum(prob * torch.log(prob + 1e-12), dim=-1))
        else:
            min_entropy = torch.mean(-torch.sum(prob * torch.log(prob + 1e-12), dim=-1) * w)
        return min_entropy
