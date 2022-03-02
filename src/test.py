
import torch
from torch import nn
import numpy as np

class convmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
        self.linear = nn.Linear(32*10*10, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x.view(x.size(0), -1))
        return x

import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

if __name__=='__main__':
    model = convmodel()

    for m in model.parameters():
        m.data.fill_(0.1)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    model.train()
    # 模拟输入8个 sample，每个的大小是 10x10，
    # 值都初始化为1，让每次输出结果都固定，方便观察
    images = torch.ones(8, 3, 10, 10)
    targets = torch.ones(8, dtype=torch.float)

    output = model(images)
    print(output.shape)
    # torch.Size([8, 20])

    loss = criterion(output.view(-1,), targets)

    print(model.conv1.weight.grad)
    # None
    loss.backward()
    print(model.conv1.weight.grad[0][0][0])
    # tensor([-0.0782, -0.0842, -0.0782])
    # 通过一次反向传播，计算出网络参数的导数，
    # 因为篇幅原因，我们只观察一小部分结果

    print(model.conv1.weight[0][0][0])
    # tensor([0.1000, 0.1000, 0.1000], grad_fn=<SelectBackward>)
    # 我们知道网络参数的值一开始都初始化为 0.1 的

    optimizer.step()
    print(model.conv1.weight[0][0][0])
    # tensor([0.1782, 0.1842, 0.1782], grad_fn=<SelectBackward>)
    # 回想刚才我们设置 learning rate 为 1，这样，
    # 更新后的结果，正好是 (原始权重 - 求导结果) ！

    optimizer.zero_grad()
    print(model.conv1.weight.grad[0][0][0])
    # tensor([0., 0., 0.])
    # 每次更新完权重之后，我们记得要把导数清零啊，
    # 不然下次会得到一个和上次计算一起累加的结果。
    # 当然，zero_grad() 的位置，可以放到前边去，
    # 只要保证在计算导数前，参数的导数是清零的就好。

    print('>>>test for bn<<<')
    bn = nn.BatchNorm2d(2)
    aa = torch.randn(2,2,1,1)
    bb = bn(aa)

    print('aa=', aa)
    print('bb=', bb)

    cc = BatchNorm(2, 4)(aa)
    print('cc=', cc)

    shape = (1, 2, 1, 1)
    mean = aa.mean(dim=(0,2,3), keepdim=True)
    dd = (aa - mean) / torch.sqrt(((aa-mean)**2).mean(dim=(0,2,3), keepdim=True))
    print('dd=', dd)

    