"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        #self.base_temperature = base_temperature
        self.base_temperature = temperature


    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).sum()

        return loss

    

'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''


class AsymSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AsymSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        #self.base_temperature = base_temperature
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='sum'):  # target_labels only from data stream
        #assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels: # target_labels only from data stream
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()            
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
    
    
class SimCLR(nn.Module):
    def __init__(self, tau=1.0, normalize=False):
        super(SimCLR, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, x, y):
        #x = torch.cat((xi, xj), dim=0)
        xi = x[:,0,:]
        xj = x[:,1,:]
        x = torch.cat((xi, xj), dim=0)

        
        #print(self.normalize)
        #print(x.shape)
        #print( (x[0,:]**2).sum() )
        #input()
        
        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.sum(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss
    
    
    
SMALL_NUM = np.log(1e-45)

class DCL(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, features, y):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        
        z1 = features[:,0,:]
        z2 = features[:,1,:]
        
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).sum()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)
        
    
class DSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        #self.base_temperature = base_temperature
        self.base_temperature = temperature
        
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


            
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

            
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        #print('mask')
        #print(mask)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        logits_mask -= mask

        exp_logits = torch.exp(logits) * logits_mask #* ((logits_mask.shape[0]-1)/logits_mask.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).sum()

        return loss

        
    
    
class DAsymSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DAsymSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        #self.base_temperature = base_temperature
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='sum'):  # target_labels only from data stream
        #assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        logits_mask -= mask
        
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels: # target_labels only from data stream
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()            
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
    
    
    
class OVA_Loss(nn.Module):
    def __init__(self):
        super(OVA_Loss, self).__init__()

    def forward(self,logits_open, label):
        logits_open = logits_open.view(logits_open.size(0), 2, -1)
        logits_open = F.softmax(logits_open, 1)
        label_s_sp = torch.zeros((logits_open.size(0),
                                  logits_open.size(2))).long().to(label.device)
        label_range = torch.arange(0, logits_open.size(0)).long()
        label_s_sp[label_range, label] = 1
        label_sp_neg = 1 - label_s_sp
        
        #print('label_s_sp:', label_s_sp)
        #print()
        
        open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                    + 1e-8) * label_s_sp, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                        + 1e-8) * label_sp_neg, 1)[0])
        Lo = open_loss_neg + open_loss
        return Lo
    
    
class OVA_Ent_Loss(nn.Module):
    def __init__(self):
        super(OVA_Ent_Loss, self).__init__()

    def forward(self, logits_open):
        logits_open = logits_open.view(logits_open.size(0), 2, -1)
        logits_open = F.softmax(logits_open, 1)
        Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                       torch.log(logits_open + 1e-8), 1), 1))
        return Le

    
class SOCR_Loss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SOCR_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        #self.base_temperature = base_temperature
        self.base_temperature = temperature

    def forward(self, logits_1, logits_2, reduction='sum'):  # target_labels only from data stream
        #assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        logits_open_u1 = logits_1.view(logits_1.size(0), 2, -1)
        logits_open_u2 = logits_2.view(logits_2.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
            logits_open_u1 - logits_open_u2)**2, 1), 1))        

        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                    logits_open_u1 - logits_open_u2)**2, 1), 1))
        return L_socr
    
    
    