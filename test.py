import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import torchvision
import models
import voc_loader
import loss
# import visualize


batch_size = 1
learning_rate = 1e-10
epoch_num = 30
best_test_loss = np.inf
pretrained = 'reload'
use_cuda = torch.cuda.is_available()
path = os.path.expanduser('/home/yxk/Downloads/')
n_class = 21

print('load data....')
train_data = voc_loader.VOC2012ClassSeg(root=path, split='train', transform=True)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=5)
val_data = voc_loader.VOC2012ClassSeg(root=path,
                            split='val',
                            transform=True)
val_loader = torch.utils.data.DataLoader(val_data,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=5)

print('load model.....')
vgg_model = models.VGGNet(requires_grad=True)
fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

if use_cuda:
    fcn_model.cuda()

print("begin loss")
criterion = loss.CrossEntropy2d()
# loss.CrossEntropy2d()

# create your optimizer

print("begin optimizer")
optimizer = torch.optim.SGD(fcn_model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)

print('begin to train....')

# input = torch.autograd.Variable(torch.randn(1, 3, 320, 320))
# # each element in target has to have 0 <= value < nclasses
# label = torch.autograd.Variable(torch.LongTensor(1, 320, 320).random_(0, 4))
# out = fcn_model(input)
# loss = criterion(out, label)
# print(loss)

# for batch, (imgs, labels) in enumerate(train_loader):
#     print(imgs)
#     print(labels)


for batch_idx, (imgs, labels) in enumerate(train_loader):
    N = imgs.size(0)
    optimizer.zero_grad()
    print(N)
    print(labels.size())

    # labels = labels//100

    if use_cuda:
        imgs = imgs.cuda()
        labels = labels.cuda()

    img = Variable(imgs)
    label = Variable(labels)

    target = torch.autograd.Variable(torch.cuda.LongTensor(1, 320, 320).random_(0, 4))

    print(img.size())
    print(label.size())

    out = fcn_model(img)

    print("out_size", out.size())
    print("label_size", label.size())

    loss = criterion(out, label)
    # print("loss: ", loss.data[0])

    loss /= N
    loss.backward()
    optimizer.step()
    # print('loss', loss.data[0])

