import math
from torch.nn import functional as F
import torch.nn as nn
import torch

def conv_bn_act(in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv2d(in_, out_, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(out_, eps, momentum),
        Swish()
    )


class SamePadConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

#Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#全连接操作
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

#注意力机制模块
class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

#DropConnect连接
class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()

#MBConv卷积
class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )
        self.skip = skip and (stride == 1) and (in_ == out_)
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x

#MBConv模块
class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

#Efficient B0结构
class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=2):
        super().__init__()
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2, bias=False)
        #共16次MBConv操作
        self.blocks1 = MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        self.blocks2= MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.blocks3=MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.blocks4=MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.blocks5 =MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.blocks6 =MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate)
        self.blocks7 =MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)


        self.head = nn.Sequential(
            *conv_bn_act(renew_ch(320), renew_ch(1280), kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(renew_ch(1280), num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x1=self.blocks1(stem)
        x2=self.blocks2(x1)
        x3 = self.blocks3(x2)
        x4 = self.blocks4(x3)
        x5 = self.blocks5(x4)
        x6 = self.blocks6(x5)
        x7 = self.blocks7(x6)
        return [x2,x3,x5,x7]#返回4层特征


class bifpn(nn.Module):
    def __init__(self):
        super(bifpn, self).__init__()
        self.relu= nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv320 = nn.Conv2d(in_channels=320, out_channels=24, kernel_size=1, padding=0, stride=1)
        self.conv112 = nn.Conv2d(in_channels=112, out_channels=24, kernel_size=1, padding=0, stride=1)
        self.conv40 = nn.Conv2d(in_channels=40, out_channels=24, kernel_size=1, padding=0, stride=1)
        self.conv24 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=1, padding=0, stride=1)
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1)
        self.conv = nn.Conv2d(in_channels=24, out_channels=2, kernel_size=3, padding=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)

    def forward(self,efficient):
        #输入归为同通道数
        p1 = self.conv24(efficient[0])
        p2 = self.conv40(efficient[1])
        p3 = self.conv112(efficient[2])
        p4 = self.conv320(efficient[3])
        #自顶向下结构
        p3_1=self.conv1(p3+self.upsample(p4))
        p2_1 = self.conv1(p2 + self.upsample(p3_1))
        p1_1 = self.conv1(p1 + self.upsample(p2_1))
        #自底向上结构和跳连结构
        p1=p1_1
        p2 = self.conv1(p2_1+nn.functional.interpolate(p1,scale_factor=0.5,recompute_scale_factor=True)+p2)
        p3 = self.conv1(p3_1+nn.functional.interpolate(p2,scale_factor=0.5,recompute_scale_factor=True)+p3)
        p4 = self.conv1(nn.functional.interpolate(p3,scale_factor=0.5,recompute_scale_factor=True) + p4)
        #特征融合
        f1 = self.conv2(p1)
        f2 = self.conv1(p2)
        f2 = self.upsample(f2)
        f2 = self.conv2(f2)
        f3 = self.conv1(p3)
        f3 = self.upsample(f3)
        f3 = self.conv1(f3)
        f3 = self.upsample(f3)
        f3 = self.conv2(f3)
        f4 = self.conv1(p4)
        f4 = self.upsample(f4)
        f4 = self.conv1(f4)
        f4 = self.upsample(f4)
        f4 = self.conv1(f4)
        f4 = self.upsample(f4)
        f4 = self.conv2(f4)
        #softmax输出
        sum = self.conv3(f4+f3+f2+f1)
        sum = self.upsample1(sum)
        sum = self.upsample4(sum)
        sum = self.conv4(sum)
        out = self.softmax(self.conv(sum))
        return out


#以EfficietNet为backbone，以BiFPN为neck的EfficientDet网络
class EfficientDet(nn.Module):
    def __init__(self):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet(1,1)
        self.neck = bifpn()

    def forward(self,x):
        x=self.backbone(x)
        x=self.neck(x)
        return x