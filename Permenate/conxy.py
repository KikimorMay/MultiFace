from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


class Conxy(Module):
    def __init__(self, channels, num_sphere):
        super(Conxy, self).__init__()
        self.num_sphere = num_sphere
        self.each_channel = channels//num_sphere
        self.fcs = []

        for i in range(self.num_sphere):
            fc = Linear(self.each_channel, 1, bias=False)
            setattr(self, 'fc%i' % i, fc)
            self.fcs.append(fc)


    def forward(self, features,  step):
        batch_num, _ = features.shape
        vars_l = []
        for i in range(self.num_sphere):
            vars_l.append(self.fcs[i](features[:, i*self.each_channel: (i+1)*self.each_channel]))
        means = []
        stds = []
        for i in range(self.num_sphere):
            means.append(torch.mean(vars_l[i]))
            stds.append(torch.std(vars_l[i], axis=0))
        count = 0
        cors = 0
        for i in range(self.num_sphere):
            j = i+1
            while j < self.num_sphere:
                cor = torch.mm((vars_l[i] - means[i]).reshape(1, -1), vars_l[j] - means[j])/((batch_num - 1)*stds[i]*stds[j])
                cors = cors + cor
                count = count + 1
                j = j+1
        return cors/count


