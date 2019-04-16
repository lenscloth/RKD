import torch
import torch.nn as nn
import torch.nn.functional as F

from metric.utils import pdist

BIG_NUMBER = 1e12
__all__ = ['AllPairs', 'HardNegative', 'SemiHardNegative', 'DistanceWeighted', 'RandomNegative']


def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) * \
               (1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device))

    return pos_mask, neg_mask


class _Sampler(nn.Module):
    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        raise NotImplementedError


class AllPairs(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()

            apns = []
            for pair_idx in pos_pair_idx:
                anchor_idx = pair_idx[0]
                neg_indices = neg_mask[anchor_idx].nonzero()

                apn = torch.cat((pair_idx.unsqueeze(0).repeat(len(neg_indices), 1), neg_indices), dim=1)
                apns.append(apn)
            apns = torch.cat(apns, dim=0)
            anchor_idx = apns[:, 0]
            pos_idx = apns[:, 1]
            neg_idx = apns[:, 2]

        return anchor_idx, pos_idx, neg_idx


class RandomNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]
            neg_index = torch.multinomial(neg_mask.float()[anchor_idx], 1).squeeze(1)

        return anchor_idx, pos_idx, neg_index


class HardNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            pos_mask, neg_mask = pos_neg_mask(labels)
            dist = self.dist_func(embeddings)

            pos_pair_index = pos_mask.nonzero()
            anchor_idx = pos_pair_index[:, 0]
            pos_idx = pos_pair_index[:, 1]

            neg_dist = (neg_mask.float() * dist)
            neg_dist[neg_dist <= 0] = BIG_NUMBER
            neg_idx = neg_dist.argmin(dim=1)[anchor_idx]

        return anchor_idx, pos_idx, neg_idx


class SemiHardNegative(_Sampler):
    def forward(self, embeddings, labels):
        with torch.no_grad():
            dist = self.dist_func(embeddings)
            pos_mask, neg_mask = pos_neg_mask(labels)
            neg_dist = dist * neg_mask.float()

            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            tiled_negative = neg_dist[anchor_idx]
            satisfied_neg = (tiled_negative > dist[pos_mask].unsqueeze(1)) * neg_mask[anchor_idx]
            """
            When there is no negative pair that its distance bigger than positive pair, 
            then select negative pair with largest distance.
            """
            unsatisfied_neg = (satisfied_neg.sum(dim=1) == 0).unsqueeze(1) * neg_mask[anchor_idx]

            tiled_negative = (satisfied_neg.float() * tiled_negative) - (unsatisfied_neg.float() * tiled_negative)
            tiled_negative[tiled_negative == 0] = BIG_NUMBER
            neg_idx = tiled_negative.argmin(dim=1)

        return anchor_idx, pos_idx, neg_idx


class DistanceWeighted(_Sampler):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4
    """
    Distance Weighted loss assume that embeddings are normalized py 2-norm.
    """

    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (pdist(embeddings, squared=True) + torch.eye(embeddings.size(0), device=embeddings.device, dtype=torch.float32)).sqrt()
            dist = dist.clamp(min=self.cut_off)

            log_weight = ((2.0 - d) * dist.log() - ((d - 3.0)/2.0) * (1.0 - 0.25 * (dist * dist)).log())
            weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
            weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

            weight = weight + ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask).float()
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx
