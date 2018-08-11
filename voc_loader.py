#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import cv2
import random


"""
https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py
"""


class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])


    def __init__(self, root, split='train', transform=True):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        # dataset_dir = osp.join(self.root, 'VOC2007')

        self.files = collections.defaultdict(list)
        for split_file in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split_file)
            for img_name in open(imgsets_file):
                img_name = img_name.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % img_name)
                lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % img_name)
                self.files[split_file].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]    # 数据
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.uint8)

        lbl[lbl == 255] = 0
        # augment
        img, lbl = self.randomFlip(img, lbl)
        img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]          # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)   # whc -> cwh
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)   # cwh -> whc
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]          # BGR -> RGB
        lbl = lbl.numpy()
        return img, lbl

    def randomFlip(self, img, label):
        if random.random() < 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img, label

    def resize(self, img, label, s=320):
        # print(s, img.shape)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (s, s), interpolation=cv2.INTER_NEAREST)
        return img, label

    def randomCrop(self, img, label):
        h, w, _ = img.shape
        short_size = min(w, h)
        rand_size = random.randrange(int(0.7 * short_size), short_size)
        x = random.randrange(0, w - rand_size)
        y = random.randrange(0, h - rand_size)

        return img[y:y + rand_size, x:x + rand_size], label[y:y + rand_size, x:x + rand_size]
    # data augmentaion
    def augmentation(self, img, lbl):
        img, lbl = self.randomFlip(img, lbl)
        img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)
        return img, lbl

    # elif not self.predict: # for batch test, this is needed
    #     img, label = self.randomCrop(img, label)
    #     img, label = self.resize(img, label, VOCClassSeg.img_size)
    # else:
    #     pass


class VOC2012ClassSeg(VOCClassSegBase):

    # url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


"""
vocbase = VOC2012ClassSeg(root="/home/yxk/Downloads/")

print(vocbase.__len__())
img, lbl = vocbase.__getitem__(0)
img = img[:, :, ::-1]
img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
print(np.shape(img))
print(np.shape(lbl))

"""


