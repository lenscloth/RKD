import os
import tarfile

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url, check_integrity


__all__ = ['CUB2011Metric']


class CUB2011(ImageFolder):
    image_folder = 'CUB_200_2011/images'
    base_folder = 'CUB_200_2011/'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    checklist = [
        ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', '4c84da568f89519f84640c54b7fba7c2'],
        ['002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg', 'e7db63424d0e384dba02aacaf298cdc0'],
        ['198.Rock_Wren/Rock_Wren_0001_189289.jpg', '487d082f1fbd58faa7b08aa5ede3cc00'],
        ['200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg', '96fd60ce4b4805e64368efc32bf5c6fe']
    ]

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        if download:
            download_url(self.url, root, self.filename, self.tgz_md5)

            if not self._check_integrity():
                cwd = os.getcwd()
                tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
                os.chdir(root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        super(CUB2011, self).__init__(os.path.join(root, self.image_folder),
                                      transform=transform,
                                      target_transform=target_transform)

    def _check_integrity(self):
        for f, md5 in self.checklist:
            fpath = os.path.join(self.root, self.image_folder, f)
            if not check_integrity(fpath, md5):
                return False
        return True


class CUB2011Classification(CUB2011):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download)

        with open(os.path.join(root, self.base_folder, 'images.txt'), 'r') as f:
            id_to_image = [l.split(' ')[1].strip() for l in f.readlines()]

        with open(os.path.join(root, self.base_folder, 'train_test_split.txt'), 'r') as f:
            id_to_istrain = [int(l.split(' ')[1]) == 1 for l in f.readlines()]

        train_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if id_to_istrain[idx]]
        test_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if not id_to_istrain[idx]]

        if train:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in train_list]
        else:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in test_list]
        self.imgs = self.samples


class CUB2011Metric(CUB2011):
    num_training_classes = 100

    def __init__(self, root, train=False, split='none', transform=None, target_transform=None, download=False):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download)

        if train:
            if split == 'train':
                self.classes = self.classes[:(self.num_training_classes-20)]
            elif split == 'val':
                self.classes = self.classes[(self.num_training_classes-20):self.num_training_classes]
            else:
                self.classes = self.classes[:self.num_training_classes]
        else:
            self.classes = self.classes[self.num_training_classes:]

        self.class_to_idx = {cls_name: cls_ind for cls_name, cls_ind in self.class_to_idx.items()
                             if cls_name in self.classes}
        self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                        if cls_ind in self.class_to_idx.values()]
        self.imgs = self.samples
