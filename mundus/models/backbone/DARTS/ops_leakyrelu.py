""" Operations """
import torch
import torch.nn as nn
from torch.autograd import Variable
import mundus.models.backbone.DARTS.genotypes as gt
# from mundus.models.backbone.DARTS.Pawarelayer import Partial_aware_layer_improved, Partial_aware_layer_or

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'avg_pool_3x1': lambda C, stride, affine: PoolBN('avg', C, (3, 1), stride, (1, 0), affine=affine),
    'max_pool_3x1': lambda C, stride, affine: PoolBN('max', C, (3, 1), stride, (1, 0), affine=affine),
    'sep_conv_3x1': lambda C, stride, affine: SepConv(C, C, (3, 1), stride, (1, 0), affine=affine),
    'sep_conv_5x1': lambda C, stride, affine: SepConv(C, C, (5, 1), stride, (2, 0), affine=affine),
    'sep_conv_7x1': lambda C, stride, affine: SepConv(C, C, (7, 1), stride, (3, 0), affine=affine),
    'sep_conv_15x1': lambda C, stride, affine: SepConv(C, C, (15, 1), stride, (7, 0), affine=affine),
    'sep_conv_17x1': lambda C, stride, affine: SepConv(C, C, (17, 1), stride, (8, 0), affine=affine),
    'sep_conv_33x3': lambda C, stride, affine: SepConv(C, C, (33, 3), stride, (16, 1), affine=affine),
    'dil_conv_3x1': lambda C, stride, affine: DilConv(C, C, (3, 1), stride, (2, 0), 2, affine=affine),  # 5x5
    'dil_conv_5x1': lambda C, stride, affine: DilConv(C, C, (5, 1), stride, (4, 0), 2, affine=affine),  # 9x9
    'dil_conv_7x1': lambda C, stride, affine: DilConv(C, C, (7, 1), stride, (6, 0), 2, affine=affine),  # 5x5
    'dil_conv_9x1': lambda C, stride, affine: DilConv(C, C, (9, 1), stride, (8, 0), 2, affine=affine),  # 9x9
    'dil_conv_11x1': lambda C, stride, affine: DilConv(C, C, (11, 1), stride, (10, 0), 2, affine=affine),  # 9x9
    'max_pool_1x3': lambda C, stride, affine: PoolBN('max', C, (1, 3), stride, (0, 1), affine=affine),
    'sep_conv_1x3': lambda C, stride, affine: SepConv(C, C, (1, 3), stride, (0, 1), affine=affine),
    'sep_conv_1x5': lambda C, stride, affine: SepConv(C, C, (1, 5), stride, (0, 2), affine=affine),
    'sep_conv_1x7': lambda C, stride, affine: SepConv(C, C, (1, 7), stride, (0, 3), affine=affine),
    'sep_conv_1x15': lambda C, stride, affine: SepConv(C, C, (1, 15), stride, (0, 7), affine=affine),
    'sep_conv_1x17': lambda C, stride, affine: SepConv(C, C, (1, 17), stride, (0, 8), affine=affine),
    'dil_conv_1x3': lambda C, stride, affine: DilConv(C, C, (1, 3), stride, (0, 2), 2, affine=affine),  # 5x5
    'dil_conv_1x5': lambda C, stride, affine: DilConv(C, C, (1, 5), stride, (0, 4), 2, affine=affine),  # 9x9
    'dil_conv_1x7': lambda C, stride, affine: DilConv(C, C, (1, 7), stride, (0, 6), 2, affine=affine),  # 5x5
    'dil_conv_1x9': lambda C, stride, affine: DilConv(C, C, (1, 9), stride, (0, 8), 2, affine=affine),  # 9x9
    'dil_conv_1x11': lambda C, stride, affine: DilConv(C, C, (1, 11), stride, (0, 10), 2, affine=affine),  # 9x9
    'STFT_ATT_200_3x3': lambda C, stride, affine: STFT_ATT(stride=stride, affine=affine),
}

"""
def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x
"""

class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    Elu - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    Elu - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    Elu - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.elu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.elu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduce_Timewise(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.elu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, (4, 1), stride=(2, 1), padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, (4, 1), stride=(2, 1), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.elu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights, debug=False):
        """
        Args:
            x: input
        
            weights: weight for each operation
        """
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        if debug:
            for w, op in zip(weights,self._ops):
                print('----------------------------------')
                print(op)
                print('size of input feature map {}'.format(x.size()))
                print(op(x).size())
                print('----------------------------------')
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Single_Path_Op(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights, debug=False):
        """
        Args:
            x: input
        
            weights: weight for each operation
        """
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        if debug:
            for w, op in zip(weights,self._ops):
                print('----------------------------------')
                print(op)
                print('size of input feature map {}'.format(x.size()))
                print(op(x).size())
                print('----------------------------------')
        return [op(x) for _, op in zip(weights, self._ops)][torch.argmax(weights)]


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


class STFT_ATT(nn.Module):
    def __init__(self, stride=1, affine=True):
        super().__init__()
        self.stride = stride
        self.net = nn.Sequential(
            DilConv(1, 1, 3, 1, 1, dilation=1, affine=affine),
            # nn.Conv2d(1, 1, 3, 1, 1)
        )

    def forward(self, x, debug=False):
        batchsize, channel_in, time_len, Echanel = x.size()
        if debug:
            print('original input size')
            print(x.size())
        x = x.view(x.size()[0], x.size()[1], -1)
        x = x.view(-1, x.size()[-1])
        timedomain = x.size()[-1]
        # print(x.size())
        if debug:
            print('after stft input size')
            print(x.size())
        out = torch.stft(x, 200, hop_length=20, win_length=200,
                         window=torch.hann_window(200).cuda(), center=True, pad_mode='reflect', normalized=True, onesided=True,
                         return_complex=False)
        if debug:
            print('after stft size')
            print(out.size())
        real = out[:, :, :, 0].unsqueeze(1)
        imaginary = out[:, :, :, 1].unsqueeze(1)
        # out = out.unsqueeze(1).float()
        # print(out.size())
        real = self.net(real).squeeze(1)
        imaginary = self.net(imaginary).squeeze(1)
        out = torch.stack((real, imaginary), 3)
        if debug:
            print('after convlution size')
            print(out.size())
        ivert_out = torch.istft(out, 200, hop_length=20, win_length=200, window=torch.hann_window(200).cuda(), center=True
                                , normalized=True, onesided=True, return_complex=False)
        pad_value = timedomain - ivert_out.size()[1]
        ivert_out = torch.nn.functional.pad(ivert_out, (0, pad_value))
        ivert_out = ivert_out
        if debug:
            print('after istft size')
            print(ivert_out.size())
        ivert_out = ivert_out.view(batchsize, channel_in, -1).view(batchsize, channel_in, time_len, -1)
        if debug:
            print('after istft post proc size')
            print(ivert_out.size())
        # x[:, :, ::self.stride, ::self.stride]
        return ivert_out[:, :, ::self.stride, ::self.stride]


if __name__ == "__main__":
    
    device = 'cpu'
    """
    PAL = SepConv(32, 32, 3, 1, 1, affine=False)
    PAL.to(device)
    x = torch.randn(2, 32, 256, 128)
    out = PAL(Variable(x).to(device))
    print(out.size())
    PAL = PoolBN('avg', 32, 3, 1, 1, affine=False)
    PAL.to(device)
    x = torch.randn(2, 32, 256, 128)
    out = PAL(Variable(x).to(device))
    print(out.size())
    
    device = 'cpu'
    PAL = SepConv(19, 32, (3, 1), 1, (1, 0), affine=False)
    PAL.to(device)
    x = torch.randn(16, 19, 400, 48)
    out = PAL(Variable(x).to(device))
    print(out.size())
    # 'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    PAL = SepConv(19, 32, (17, 1), 1, (8, 0), affine=False)
    PAL.to(device)
    x = torch.randn(16, 19, 400, 48)
    out = PAL(Variable(x).to(device))
    print(out.size())
    PAL = DilConv(19, 32, (3, 1), 1, (2, 0), 2, affine=False)
    PAL.to(device)
    x = torch.randn(16, 19, 400, 48)
    out = PAL(Variable(x).to(device))
    print(out.size())
    PAL = DilConv(19, 32, (5, 1), 1, (4, 0), 2, affine=False)
    PAL.to(device)
    x = torch.randn(16, 19, 400, 48)
    out = PAL(Variable(x).to(device))
    print(out.size())
    PAL = FactorizedReduce(32, 32, affine=True)
    PAL.to(device)
    x = torch.randn(16, 32, 200, 24)
    out = PAL(Variable(x).to(device))
    print(out.size())
    """
    x = torch.randn(16, 32, 200, 24)
    PAL = STFT_ATT(19, 32, (3, 1), 1, (2, 0), 2)
    PAL.to(device)
    out = PAL(Variable(x).to(device))
    print(out.size())

