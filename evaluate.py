import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import models
import voc_loader
import numpy as np
from torch.autograd import Variable
import tools


n_class = 21
def evaluate():
    use_cuda = torch.cuda.is_available()
    path = os.path.expanduser('/home/yxk/data/')
    val_data = voc_loader.VOC2012ClassSeg(root=path,
                                split='val',
                                transform=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=5)
    print('load model .....')
    vgg_model = models.VGGNet(requires_grad=True)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)
    fcn_model.load_state_dict(torch.load('params.pth'))

    if use_cuda:
        fcn_model.cuda()
    fcn_model.eval()

    label_trues, label_preds = [], []
    # for idx, (img, label) in enumerate(val_loader):
    for idx in range(len(val_data)):
        img, label = val_data[idx]
        img = img.unsqueeze(0)
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        out = fcn_model(img)     # 1, 21, 320, 320

        pred = out.data.max(1)[1].squeeze_(1).squeeze_(0)   # 320, 320

        if use_cuda:
            pred = pred.cpu()
        label_trues.append(label.numpy())
        label_preds.append(pred.numpy())

        if idx % 30 == 0:
            print('evaluate [%d/%d]' % (idx, len(val_loader)))

    metrics = tools.accuracy_score(label_trues, label_preds)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))


if __name__ == '__main__':
    evaluate()