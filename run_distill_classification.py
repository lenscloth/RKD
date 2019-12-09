'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model.backbone.classification import *
from model.utils import progress_bar
from metric.loss_classification import DistillAngle, DistillRelativeDistanceV2, DarkKnowledge


LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--student',
                    choices=dict(resnet18=ResNet18,
                                 resnet50=ResNet50,
                                 vgg16bn=VGG16bn,
                                 vgg11bn=VGG11bn),
                    action=LookupChoices)
parser.add_argument('--teacher',
                    choices=dict(resnet18=ResNet18,
                                 resnet50=ResNet50),
                    action=LookupChoices)
parser.add_argument('--data',
                    required=True,
                    choices=dict(cifar10=torchvision.datasets.CIFAR10,
                                 cifar100=torchvision.datasets.CIFAR100),
                    action=LookupChoices)
parser.add_argument('--dist_ratio', default=25, type=float)
parser.add_argument('--kd_ratio', default=16, type=float)
parser.add_argument('--angle_ratio', default=50, type=float)
parser.add_argument('--load', default=None, required=True)
parser.add_argument('--save', default='checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = args.data(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = args.data(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
student = args.student(num_classes = 100 if isinstance(trainset, torchvision.datasets.CIFAR100) else 10)
teacher = args.teacher(num_classes = 100 if isinstance(trainset, torchvision.datasets.CIFAR100) else 10)

# Load teacher.
print('==> Load teacher..')
assert os.path.isdir(args.load), 'Error: no checkpoint directory found!'
checkpoint = torch.load('%s/ckpt.t7'%args.load)
teacher.load_state_dict(checkpoint['net'])
best_acc = 0
start_epoch = 0

print(student)
student = student.to(device)
teacher = teacher.to(device)
teacher.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


dist_criterion = DistillRelativeDistanceV2()
angle_criterion = DistillAngle()
kd_criterion = DarkKnowledge(temp=4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    student.train()
    train_cls_loss = 0
    train_dist_loss = 0
    train_angle_loss = 0
    train_kd_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, e_s = student(inputs, True)
        with torch.no_grad():
            teacher_output, e_t = teacher(inputs, True)

        cls_loss = criterion(outputs, targets)
        norm_es, norm_et = F.normalize(e_s, p=2, dim=1), F.normalize(e_t, p=2, dim=1)
        dist_loss = args.dist_ratio * dist_criterion(norm_es, norm_et)
        angle_loss = args.angle_ratio * angle_criterion(norm_es, norm_et)
        kd_loss = args.kd_ratio * F.kl_div(F.log_softmax(outputs/4, dim=1), F.softmax(teacher_output/4, dim=1), reduction='none').sum(dim=1).mean()
        loss = cls_loss + dist_loss + angle_loss + kd_loss
        loss.backward()
        optimizer.step()

        train_cls_loss += cls_loss.item()
        train_dist_loss += dist_loss.item()
        train_angle_loss += angle_loss.item()
        train_kd_loss += kd_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Cls Loss: %.3f | Dist Loss: %.5f | Angle Loss: %.5f | KD Loss: %.5f | Acc: %.3f%% (%d/%d)'
            % (train_cls_loss/(batch_idx+1), (train_dist_loss/(batch_idx+1)), (train_angle_loss/(batch_idx+1)),
               (train_kd_loss/(batch_idx+1)), 100.*correct/total, correct, total))


def test(net, epoch, save=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if save:
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': student.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
            torch.save(state, './%s/ckpt.t7' % args.save)
            best_acc = acc
        print("Best Accuracy %.2f" % best_acc)

test(teacher, 0, save=False)
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(student, epoch, save=True)
