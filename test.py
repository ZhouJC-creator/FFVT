import os

import imageio
import numpy as np
from PIL import Image


class cotton():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.mode = 'train' if is_train else 'test'
        anno_txt_file = open(os.path.join(self.root, 'anno', self.mode + '.txt'))
        self.labels = []
        self.imgs_name = []
        for line in anno_txt_file:
            self.imgs_name.append(line.strip().split(' ')[0])
            self.labels.append(int(line.strip().split(' ')[1]))
        # self.imgs = [scipy.misc.imread(os.path.join(self.root, 'images', img_name)) for img_name in self.imgs_name ]

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images', self.imgs_name[index])
        img, target, imgname = imageio.imread(img_path), self.labels[index], self.imgs_name[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs_name)


cotton = cotton('F:/DataSet/ip_102')
img, target = cotton.__getitem__(1)
print(img)
print(target)
