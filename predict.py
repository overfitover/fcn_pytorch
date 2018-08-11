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
                               split='train',
                               transform=True)

    vgg_model = models.VGGNet(requires_grad=True)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    fcn_model.load_state_dict(torch.load('params.pth'))

    fcn_model.eval()

    if use_cuda:
        fcn_model.cuda()

    criterion = CrossEntropy2d()

    for i in range(len(dataset)):
        idx = random.randrange(0, len(dataset))
        img, label = dataset[idx]
        img_name = str(i)

        img_src, _ = dataset.untransform(img, label)    # whc

        cv2.imwrite(path + 'image/%s_src.jpg' % img_name, img_src)
        tools.labelTopng(label, path + 'image/%s_label.png' % img_name)  # 将label转换成图片

        print(img_name)

        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img.unsqueeze(0), volatile=True)
        label = Variable(label.unsqueeze(0), volatile=True)

        out = fcn_model(img)
        loss = criterion(out, label)
        print('loss:', loss.data[0])

        label = out.data.max(1)[1].squeeze_(1).squeeze_(0)    # 320, 320
        if use_cuda:
            label = label.cpu()

        tools.labelTopng(label, path + 'image/%s_out.png' % img_name)   # 将网络输出转换成图片

        if i == 10:
            break


if __name__ == '__main__':
    main()