import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class DarkKnowledge(nn.Module):
    def __init__(self, temp=4):
        super().__init__()
        self.temp = temp

    def forward(self, student, teacher):
        s = F.log_softmax(student/self.temp, dim=1)
        t = F.softmax(teacher/self.temp, dim=1)

        return F.kl_div(s, t, reduction='none').sum(dim=1).mean()


class DistillDistance(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
        d = pdist(student, squared=False)
        loss = F.smooth_l1_loss(d, self.alpha * t_d, reduction='elementwise_mean')
        return loss


class DistillRelativeDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            t_d = t_d / teacher.pow(2).sum(dim=1).sqrt().mean()

        d = pdist(student, squared=False)
        d = d / student.pow(2).sum(dim=1).sqrt().mean()

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class DistillRelativeDistanceV2(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class DistillAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class DistillAttention(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(dim=1).view(student.size(0), -1), p=2, dim=1)

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(dim=1).view(teacher.size(0), -1), p=2, dim=1)

        return (s_attention - t_attention).pow(2).mean()