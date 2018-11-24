import os
import tarfile
import scipy.io as io

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, check_integrity

__all__ = ['Cars196Metric']


class Cars196Metric(ImageFolder):
    base_folder = 'car_ims'
    img_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    img_filename = 'car_ims.tgz'
    img_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

    anno_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    anno_filename = 'cars_annos.mat'
    anno_md5 = 'b407c6086d669747186bd1d764ff9dbc'

    checklist = [
        ['016185.jpg', 'bab296d5e4b2290d024920bf4dc23d07'],
        ['000001.jpg', '2d44a28f071aeaac9c0802fddcde452e'],
    ]

    test_list = []
    num_training_classes = 98

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root + "/Cars196"
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            download_url(self.img_url, self.root, self.img_filename, self.img_md5)
            download_url(self.anno_url, self.root, self.anno_filename, self.anno_md5)

            if not self._check_integrity():
                cwd = os.getcwd()
                tar = tarfile.open(os.path.join(self.root, self.img_filename), "r:gz")
                os.chdir(self.root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)

        if not self._check_integrity() or \
           not check_integrity(os.path.join(self.root, self.anno_filename), self.anno_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        ImageFolder.__init__(self, os.path.join(self.root),
                             transform=transform, target_transform=target_transform, **kwargs)
        self.root = root + "/Cars196"

        labels = io.loadmat(os.path.join(self.root, self.anno_filename))['annotations'][0]
        class_names = io.loadmat(os.path.join(self.root, self.anno_filename))['class_names'][0]

        if train:
            self.classes = [str(c[0]) for c in class_names[:98]]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.classes = [str(c[0]) for c in class_names[98:]]
            self.class_to_idx = {cls: i+98 for i, cls in enumerate(self.classes)}

        class_idx = list(self.class_to_idx.values())
        samples = []
        for l in labels:
            cls = int(l[5][0, 0]) - 1
            p = l[0][0]
            if cls in class_idx:
                samples.append((os.path.join(self.root, p), int(cls)))

        self.samples = samples
        self.imgs = self.samples

    def _check_integrity(self):
        for f, md5 in self.checklist:
            fpath = os.path.join(self.root, self.base_folder, f)
            if not check_integrity(fpath, md5):
                return False
        return True
