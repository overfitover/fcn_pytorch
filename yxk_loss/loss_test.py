import torch
from torch.autograd import Variable

'''
参考文献：　https://blog.csdn.net/zhangxb35/article/details/72464152?utm_source=itdadao&utm_medium=referral
如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss；
如果 reduce = True，那么 loss 返回的是标量

    如果 size_average = True，返回 loss.mean();
    如果 size_average = True，返回 loss.sum();
'''

# nn.L1Loss: loss(input, target)=|input-target|
if False:
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())


# nn.SmoothL1Loss 　在(-1, 1)上是平方loss, 其他情况是L1 loss
if False:
    loss_fn = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())

# nn.MSELoss  均方损失函数
if False:
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())

# nn.BCELoss
if False:
    import torch.nn.functional as F

    loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.FloatTensor(3, 4).random_(2))
    loss = loss_fn(F.sigmoid(input), target)
    print(input, input.shape)
    print(F.sigmoid(input))
    print(target, target.shape)
    print(loss, loss.shape)

# nn.CrossEntropyLoss
if False:
    weight = torch.Tensor([1, 2, 1, 1, 10])
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=None)
    input = Variable(torch.randn(3, 5))  # (batch_size, C)
    target = Variable(torch.LongTensor(3).random_(5))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)


