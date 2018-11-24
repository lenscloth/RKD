import os
import zipfile

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, check_integrity


__all__ = ['StanfordOnlineProductsMetric']


class StanfordOnlineProductsMetric(ImageFolder):
    base_folder = 'Stanford_Online_Products'
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    zip_md5 = '7f73d41a2f44250d4779881525aea32e'

    checklist = [
        ['bicycle_final/111265328556_0.JPG', '77420a4db9dd9284378d7287a0729edb'],
        ['chair_final/111182689872_0.JPG', 'ce78d10ed68560f4ea5fa1bec90206ba'],
        ['table_final/111194782300_0.JPG', '8203e079b5c134161bbfa7ee2a43a0a1'],
        ['toaster_final/111157129195_0.JPG', 'd6c24ee8c05d986cafffa6af82ae224e']
    ]
    num_training_classes = 11318

    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        if download:
            download_url(self.url, self.root, self.filename, self.zip_md5)

            if not self._check_integrity():
                # extract file
                cwd = os.getcwd()
                os.chdir(root)
                with zipfile.ZipFile(self.filename, "r") as zip:
                    zip.extractall()
                os.chdir(cwd)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        ImageFolder.__init__(self, os.path.join(root, self.base_folder),
                             transform=transform, target_transform=target_transform, **kwargs)

        self.super_classes = self.classes
        samples = []
        classes = set()
        f = open(os.path.join(root, self.base_folder, 'Ebay_{}.txt'.format('train' if train else 'test')))
        f.readline()
        for (image_id, class_id, super_class_id, path) in map(str.split, f):
            samples.append((os.path.join(root, self.base_folder, path), int(class_id)-1))
            classes.add("%s.%s" % (class_id, self.super_classes[int(super_class_id)-1]))

        self.samples = samples
        self.classes = list(classes)
        self.classes.sort(key=lambda x: int(x.split(".")[0]))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = self.samples

    def _check_integrity(self):
        for f, md5 in self.checklist:
            fpath = os.path.join(self.root, self.base_folder, f)
            if not check_integrity(fpath, md5):
                return False
        return True
