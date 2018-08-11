import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import voc_loader
import models
import random
import tools
from loss import CrossEntropy2d

n_class = 21
def main():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('/home/yxk/Downloads/')

    dataset = voc_loader.VOC2012ClassSeg(root=path,
                               split='val',
                               transform=True)

    print("begin...")
    for i in range(len(dataset)):

        idx = random.randrange(0, len(dataset))
        img, label = dataset[idx]
        img_name = str(i)
        img_src, _ = dataset.untransform(img, label)    # whc

        cv2.imwrite(path + 'image/%s_src.jpg' % img_name, img_src)
        tools.labelTopng(label, path + 'image/%s_label.png' % img_name)  # 将label转换成图片

        if i == 10:
            break


if __name__ == '__main__':
    main()