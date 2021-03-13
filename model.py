from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        # self.avg_pool = AvgPool2d(feature_shape)
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE_NEW(Module):
    def __init__(self, in_channel, depth, stride, feature_shape):
        super(bottleneck_IR_SE_NEW, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            ReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 4, feature_shape)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut




class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride', 'feature_shape'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, feature_shape, stride = 2):
  return [Bottleneck(in_channel, depth, stride, 2*feature_shape)] + [Bottleneck(depth, depth, 1, feature_shape) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, feature_shape=56, num_units=2),
            get_block(in_channel=64, depth=96, feature_shape=28, num_units=2),
            get_block(in_channel=96, depth=128, feature_shape=14, num_units=2),
            get_block(in_channel=128, depth=128, feature_shape=7, num_units=2)
        ]


        # 3M :
        # blocks = [
        #     get_block(in_channel=64, depth=64, feature_shape=56, num_units=2),
        #     get_block(in_channel=64, depth=96, feature_shape=28, num_units=2),
        #     get_block(in_channel=96, depth=128, feature_shape=14, num_units=2),
        #     get_block(in_channel=128, depth=256, feature_shape=7, num_units=2)
        # ]  下面全连接层也是256
    if num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, feature_shape=56,num_units=3),
            get_block(in_channel=64, depth=128, feature_shape=28, num_units=4),
            get_block(in_channel=128, depth=256, feature_shape=14,num_units=14),
            get_block(in_channel=256, depth=512, feature_shape=7, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64,feature_shape=56, num_units=3),
            get_block(in_channel=64, depth=128, feature_shape=28,num_units=13),
            get_block(in_channel=128, depth=256,feature_shape=14, num_units=30),
            get_block(in_channel=256, depth=512,feature_shape=7, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir_se'):
        super(Backbone, self).__init__()
        assert num_layers in [34, 50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se', 'ir_se_new'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        elif mode == 'ir_se_work':
            unit_module = bottleneck_IR_SE_NEW

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        print('model is:', mode, 'depth is:', num_layers)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

##################################  MobileFaceNet #############################################################

class Backbone_work(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir_se'):
        super(Backbone_work, self).__init__()
        assert num_layers in [18, 50, 100, 152], 'num_layers should be 50,100, or 152'
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR_SE_NEW
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      ReLU(64))
        # self.output_layer1 = Sequential(BatchNorm2d(128),
        #                                 Conv2d(128, 128, (7, 7), 1, 0, bias=False),
        #                                 ReLU(128))
        #                                 #Dropout(drop_ratio))   # 128 * 1 * 1

        self.output_layer2 = Sequential(AvgPool2d(7, 1),
                                        BatchNorm2d(128),
                                        #Flatten(),
                                        #Linear(128, 512),
                                        Conv2d(128,512,1,1,0,bias=False),
                                        BatchNorm2d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride,
                                bottleneck.feature_shape))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        # x = self.output_layer1(x)
        x = self.output_layer2(x)
        x = x.view(x.shape[0], x.shape[1])
        return l2_norm(x)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        print('MobileFaceNet model')

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)
        # return out


class MobileFaceNetSoftmax(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNetSoftmax, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        print('MobileFaceNet Softmax model, embedding size is:', embedding_size)

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        # return l2_norm(out)
        return out



##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512,  classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        print('Arcface head')

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output


##################################  Arcface_Sphere head #############################################################

class ArcfaceMultiSphere(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, num_shpere=4, s=64., m=1/5.0):
        super(ArcfaceMultiSphere, self).__init__()
        self.classnum = classnum
        self.num_sphere = num_shpere
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.each_embeding_size = embedding_size // num_shpere

        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        print('ArcfaceMultiSphere head', 'num_sphere is:', num_shpere, 'the margin is:', m)

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        output_list = []
        for i in range(self.num_sphere):
            kernel_norm = l2_norm(self.kernel[i * self.each_embeding_size:(i + 1) * self.each_embeding_size, :], axis=0)
            cos_theta = torch.mm(embbedings[:, i * self.each_embeding_size:(i + 1) * self.each_embeding_size], kernel_norm)
            cos_theta = cos_theta.clamp(-1, 1)
            cos_theta_2 = torch.pow(cos_theta, 2)
            sin_theta_2 = 1 - cos_theta_2
            sin_theta = torch.sqrt(sin_theta_2)
            cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

            cond_v = cos_theta - self.threshold
            cond_mask = cond_v <= 0
            keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
            cos_theta_m[cond_mask] = keep_val[cond_mask]
            output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
            idx_ = torch.arange(0, nB, dtype=torch.long)
            output[idx_, label] = cos_theta_m[idx_, label]
            output *= self.s
            output_list.append(output)
        return output_list


##################################  Softmax head #############################################################

class Softmax(Module):
    def __init__(self, embedding_size=512,  classnum=51332, s=64.):
        super(Softmax, self).__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum
        self.s = s
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        print('Softmax head')


    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        out = torch.mm(embbedings, kernel_norm)
        out *= self.s
        return out

##################################  Softmax_Multi_Sphere head #############################################################
class MultiSphereSoftmax(Module):
    pass

##################################  Cosface head #############################################################
    
class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 64. # see normface https://arxiv.org/abs/1704.06369
        print('Am_softmax head', 'margin is:', self.m)

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

class MultiAm_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, num_sphere=4, m=0.35):
        super(MultiAm_softmax, self).__init__()
        self.classnum = classnum
        self.num_sphere = num_sphere

        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.kernel_list = []
        self.each_embeding_size = embedding_size // num_sphere
        self.s = 64. # see normface https://arxiv.org/abs/1704.06369

        print('MultiAm_softmax head', 'num_sphere is:', num_sphere, 'margin is:', m,'s is: ', self.s)
        self.m = m # additive margin recommended by the paper
    def forward(self, embbedings, label):
        output_list = []
        for i in range(self.num_sphere):
            kernel_norm = l2_norm(self.kernel[i * self.each_embeding_size:(i + 1) * self.each_embeding_size, :], axis=0)
            cos_theta = torch.mm(embbedings[:, i * self.each_embeding_size:(i + 1) * self.each_embeding_size],
                                 kernel_norm)

            cos_theta = cos_theta.clamp(-1,1) # for numerical stability
            phi = cos_theta - self.m
            label = label.view(-1,1) #size=(B,1)
            index = cos_theta.data * 0.0 #size=(B,Classnum)
            index.scatter_(1,label.data.view(-1,1),1)
            index = index.byte()
            output = cos_theta * 1.0
            output[index] = phi[index] #only change the correct predicted output
            output *= self.s # scale up in order to make softmax work, first introduced in normface
            output_list.append(output)
        return output_list



