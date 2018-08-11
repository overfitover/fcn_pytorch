import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# # 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 构造一个线性模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()


writer = SummaryWriter('log')
for i in range(4000):
    prediction = net(x)
    loss = loss_func(prediction, y)

    writer.add_scalar('loss', loss, i)

    # model save
    # torch.save(net, 'para.pth')
    torch.save(net.state_dict(), 'ee.pth')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 画图
    # if i%5 == 0:
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f'% loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
    #     plt.pause(0.1)

# model load
# model = torch.load('para.pth')
model = Net(n_feature=1, n_hidden=10, n_output=1)
model.load_state_dict(torch.load('ee.pth'))

a = torch.unsqueeze(torch.linspace(-1, 1, 4), dim=1)
print(a**2)
b = model(a)
print(b)