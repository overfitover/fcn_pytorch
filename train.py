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
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from argparse import ArgumentParser


# argumentparse
parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=2, help="batch size of the data")
parser.add_argument('-e', '--epochs', type=int, default=10, help='epoch of the train')
parser.add_argument('-c', '--n_class', type=int, default=21, help='the classes of the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

# import visualize
writer = SummaryWriter()

batch_size = args.batch_size
learning_rate = args.learning_rate
epoch_num = args.epochs
n_class = args.n_class


best_test_loss = np.inf
pretrained = 'reload'
use_cuda = torch.cuda.is_available()

path = os.path.expanduser('/home/yxk/Downloads/')

# dataset 2007
# path = os.path.expanduser('/home/yxk/project/')

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

vgg_model = models.VGGNet(requires_grad=True)
fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

if use_cuda:
    fcn_model.cuda()

criterion = loss.CrossEntropy2d()
# create your optimizer
optimizer = Adam(fcn_model.parameters())
# optimizer = torch.optim.SGD(fcn_model.parameters(), lr=0.01)

def train(epoch):
    fcn_model.train()
    total_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()

        imgs = Variable(imgs)
        labels = Variable(labels)

        out = fcn_model(imgs)

        loss = criterion(out, labels)
        loss /= N

        # visiualize scalar
        writer.add_scalar("loss", loss, batch_idx)
        writer.add_scalar("total_loss", total_loss, batch_idx)
        writer.add_scalars('loss/scalar_group', {"loss": batch_idx*loss,
                                                 "total_loss": batch_idx*total_loss})
        writer.add_image('Image', imgs, batch_idx)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]  # return float

        if (batch_idx) % 20 == 0:
            print('train epoch [%d/%d], iter[%d/%d], lr %.5f, aver_loss %.5f' % (epoch,
                                                                                 epoch_num, batch_idx,
                                                                                 len(train_loader), learning_rate,
                                                                                 total_loss / (batch_idx + 1)))

        # model save
        if (epoch) % 5 == 0:
            torch.save(fcn_model.state_dict(), 'params.pth')

        assert total_loss is not np.nan
        assert total_loss is not np.inf

    total_loss /= len(train_loader)
    print('train epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))


def test(epoch):
    fcn_model.eval()
    total_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        imgs = Variable(imgs)    # , volatile=True
        labels = Variable(labels)  # , volatile=True
        out = fcn_model(imgs)
        loss = criterion(out, labels)
        loss /= N
        total_loss += loss.data[0]

        if (batch_idx + 1) % 3 == 0:
            print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch,
                                                                       epoch_num, batch_idx, len(val_loader),
                                                                       total_loss / (batch_idx + 1)))



    total_loss /= len(val_loader)
    print('test epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))

    global best_test_loss
    if best_test_loss > total_loss:
        best_test_loss = total_loss
        print('best loss....')
        # fcn_model.save('SBD.pth')


if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
        # test(epoch)

        # adjust learning rate
        if epoch == 1 or epoch == 2:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate
            # optimizer.param_groups[1]['lr'] = learning_rate * 2