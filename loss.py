import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):
    '''
    loss doesn't change, loss can not be backward?
    '''
    def __init__(self):
        super(CrossEntropy2d, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=None, size_average=True)  # should size_average=False?

    def forward(self, out, target):
        n, c, h, w = out.size()  # n:batch_size, c:class
        out = out.view(-1, c)
        target = target.view(-1)
        # print('out', out.size(), 'target', target.size())

        loss = self.criterion(out, target)

        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def smooth_l1(deltas, targets, sigma=3.0):
    """
    :param deltas: (tensor) predictions, sized [N,D].
    :param targets: (tensor) targets, sized [N,].
    :param sigma: 3.0
    :return:
    """

    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.min(torch.abs(diffs), 1.0 / sigma2).detach().float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + \
                    torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, float)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else:
            return loss.sum()










