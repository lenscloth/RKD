import os
import torch
import torch.nn as nn
import h5py

from collections import OrderedDict
from torchvision.datasets.utils import download_url

__all__ = ["GoogleNet"]


class GoogleNet(nn.Sequential):
    output_size = 1024
    input_side = 227
    rescale = 255.0
    rgb_mean = [122.7717, 115.9465, 102.9801]
    rgb_std = [1, 1, 1]
    url = "https://github.com/vadimkantorov/metriclearningbench/releases/download/data/googlenet.h5"
    md5hash = 'c7d7856bd1ab5cb02618b3f7f564e3c6'
    model_filename = 'googlenet.h5'

    def __init__(self, pretrained=True, root='data'):
        super(GoogleNet, self).__init__(OrderedDict([
            ('conv1', nn.Sequential(OrderedDict([
                ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),
                ('lrn1', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1))
            ]))),

            ('conv2', nn.Sequential(OrderedDict([
                ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
                ('relu1', nn.ReLU(True)),
                ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
                ('relu2', nn.ReLU(True)),
                ('lrn2', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True))
            ]))),

            ('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)),

            ('pool3', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)),

            ('pool4', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)),

            ('pool5', nn.AvgPool2d((7, 7), (1, 1), ceil_mode=True)),
        ]))

        if pretrained:
            self.load(root)

    def load(self, root):
        download_url(self.url, root, self.model_filename, self.md5hash)
        h5_file = h5py.File(os.path.join(root, self.model_filename), 'r')
        group_key = list(h5_file.keys())[0]
        self.load_state_dict({k: torch.from_numpy(v[group_key].value) for k, v in h5_file[group_key].items()})


class InceptionModule(nn.Module):
    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5,
                 outplane_pool_proj):
        super(InceptionModule, self).__init__()
        a = nn.Sequential(OrderedDict([
            ('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0))),
            ('1x1_relu', nn.ReLU(True))
        ]))

        b = nn.Sequential(OrderedDict([
            ('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))),
            ('3x3_relu1', nn.ReLU(True)),
            ('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1))),
            ('3x3_relu2', nn.ReLU(True))
        ]))

        c = nn.Sequential(OrderedDict([
            ('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))),
            ('5x5_relu1', nn.ReLU(True)),
            ('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2))),
            ('5x5_relu2', nn.ReLU(True))
        ]))

        d = nn.Sequential(OrderedDict([
            ('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
            ('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
            ('pool_relu', nn.ReLU(True))
        ]))

        for container in [a, b, c, d]:
            for name, module in container.named_children():
                self.add_module(name, module)

        self.branches = [a, b, c, d]

    def forward(self, input):
        return torch.cat([branch(input) for branch in self.branches], 1)
