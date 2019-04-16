import os
import argparse

import dataset
import model.backbone as backbone
import metric.pairsampler as pair

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from metric.utils import recall
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, FitNet
from model.embedding import LinearEmbedding


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset',
                    choices=dict(cub200=dataset.CUB2011Metric,
                                 cars196=dataset.Cars196Metric,
                                 stanford=dataset.StanfordOnlineProductsMetric),
                    default=dataset.CUB2011Metric,
                    action=LookupChoices)

parser.add_argument('--base',
                    choices=dict(resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--teacher_base',
                    choices=dict(resnet18=backbone.ResNet18,
                                 resnet50=backbone.ResNet50),
                    default=backbone.ResNet50,
                    action=LookupChoices)

parser.add_argument('--triplet_ratio', default=0, type=float)

parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)

parser.add_argument('--dark_ratio', default=0, type=float)
parser.add_argument('--dark_alpha', default=2, type=float)
parser.add_argument('--dark_beta', default=3, type=float)

parser.add_argument('--fitnet_ratio', default=1, type=float)

parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.AllPairs,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--embedding_size', default=128, type=int)

parser.add_argument('--teacher_load', default=None, required=True)
parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--teacher_embedding_size', default=128, type=int)

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--save_dir', default=None)

opts = parser.parse_args()
student_base = opts.base(pretrained=True)
teacher_base = opts.teacher_base(pretrained=False)


def get_normalize(net):
    google_mean = torch.Tensor([104, 117, 128]).view(1, -1, 1, 1).cuda()
    google_std = torch.Tensor([1, 1, 1]).view(1, -1, 1, 1).cuda()
    other_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    other_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()

    def googlenorm(x):
        x = x[:, [2, 1, 0]] * 255
        x = (x - google_mean) / google_std
        return x

    def othernorm(x):
        x = (x - other_mean) / other_std
        return x

    if isinstance(net, backbone.InceptionV1BN) or isinstance(net, backbone.GoogleNet):
        return googlenorm
    else:
        return othernorm


teacher_normalize = get_normalize(teacher_base)
student_normalize = get_normalize(student_base)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=True)
dataset_train_eval = opts.dataset(opts.data, train=True, transform=test_transform, download=True)
dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=True)

print("Number of images in Training Set: %d" % len(dataset_train))
print("Number of images in Test set: %d" % len(dataset_eval))

loader_train_sample = DataLoader(dataset_train, batch_sampler=NPairs(dataset_train, opts.batch, m=5,
                                                                     iter_per_epoch=opts.iter_per_epoch),
                                 pin_memory=True, num_workers=8)
loader_train_eval = DataLoader(dataset_train_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                               pin_memory=False, num_workers=8)
loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=opts.batch, drop_last=False,
                         pin_memory=True, num_workers=8)

student = LinearEmbedding(student_base,
                          output_size=student_base.output_size,
                          embedding_size=opts.embedding_size,
                          normalize=opts.l2normalize == 'true')

teacher = LinearEmbedding(teacher_base,
                          output_size=teacher_base.output_size,
                          embedding_size=opts.teacher_embedding_size,
                          normalize=opts.teacher_l2normalize == 'true')

teacher.load_state_dict(torch.load(opts.teacher_load))
student = student.cuda()
teacher = teacher.cuda()


optimizer = optim.Adam(student.parameters(), lr=opts.lr, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
fitnet_criterion = [FitNet(64, 256), FitNet(128, 512), FitNet(256, 1024), FitNet(512, 2048), FitNet(opts.embedding_size, 512)]
[f.cuda() for f in fitnet_criterion]


def train_fitnet(loader, ep):
    lr_scheduler.step()
    student.train()
    teacher.eval()
    loss_all = []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()

        b1, b2, b3, b4, pool, e = student(student_normalize(images), True)
        with torch.no_grad():
            t_b1, t_b2, t_b3, t_b4, t_pool, t_e = teacher(teacher_normalize(images), True)

        loss = opts.fitnet_ratio * (fitnet_criterion[1](b2, t_b2) +
                                    fitnet_criterion[2](b3, t_b3) +
                                    fitnet_criterion[3](b4, t_b4) +
                                    fitnet_criterion[4](e, t_e))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())

        train_iter.set_description("[Train][Epoch %d] FitNet: %.5f" % (ep, loss.item()))
    print('[Epoch %d] Loss: %.5f \n' % (ep, torch.Tensor(loss_all).mean()))


def train(loader, ep):
    lr_scheduler.step()
    student.train()
    teacher.eval()

    dist_loss_all = []
    angle_loss_all = []
    dark_loss_all = []
    triplet_loss_all = []
    loss_all = []

    train_iter = tqdm(loader)
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()

        e = student(student_normalize(images))
        with torch.no_grad():
            t_e = teacher(teacher_normalize(images))

        triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)
        dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
        angle_loss = opts.angle_ratio * angle_criterion(e, t_e)
        dark_loss = opts.dark_ratio * dark_criterion(e, t_e)

        loss = triplet_loss + dist_loss + angle_loss + dark_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        triplet_loss_all.append(triplet_loss.item())
        dist_loss_all.append(dist_loss.item())
        angle_loss_all.append(angle_loss.item())
        dark_loss_all.append(dark_loss.item())
        loss_all.append(loss.item())

        train_iter.set_description("[Train][Epoch %d] Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %5f" %
                                   (ep, triplet_loss.item(), dist_loss.item(), angle_loss.item(), dark_loss.item()))
    print('[Epoch %d] Loss: %.5f, Triplet: %.5f, Dist: %.5f, Angle: %.5f, Dark: %.5f \n' %\
          (ep, torch.Tensor(loss_all).mean(), torch.Tensor(triplet_loss_all).mean(),
           torch.Tensor(dist_loss_all).mean(), torch.Tensor(angle_loss_all).mean(), torch.Tensor(dark_loss_all).mean()))


def eval(net, normalize, loader, ep):
    K = [1]
    net.eval()
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            output = net(normalize(images))
            embeddings_all.append(output.data)
            labels_all.append(labels.data)
            test_iter.set_description("[Eval][Epoch %d]" % ep)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = recall(embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print('[Epoch %d] Recall@%d: [%.4f]\n' % (ep, k, 100 * r))
    return rec[0]


eval(teacher, teacher_normalize, loader_train_eval, 0)
eval(teacher, teacher_normalize, loader_eval, 0)
best_train_rec = eval(student, student_normalize, loader_train_eval, 0)
best_val_rec = eval(student, student_normalize, loader_eval, 0)

for epoch in range(1, opts.epochs+1):
    train_fitnet(loader_train_sample, epoch)
    train_recall = eval(student, student_normalize, loader_train_eval, epoch)
    val_recall = eval(student, student_normalize, loader_eval, epoch)

    if best_train_rec < train_recall:
        best_train_rec = train_recall

    if best_val_rec < val_recall:
        best_val_rec = val_recall
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

    if opts.save_dir is not None:
        if not os.path.isdir(opts.save_dir):
            os.mkdir(opts.save_dir)
        torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
        with open("%s/result.txt" % opts.save_dir, 'w') as f:
            f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))
            f.write("Best Test Recall@1: %.4f\n" % (best_val_rec * 100))
            f.write("Final Recall@1: %.4f\n" % (val_recall * 100))

    print("Best Train Recall: %.4f" % best_train_rec)
    print("Best Eval Recall: %.4f" % best_val_rec)
