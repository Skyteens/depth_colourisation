import torch
import torch.nn as nn
from .models.ResNeXt import *
from torch.nn import functional as F
import math

def lateral_resnext50_32x4d_body_stride16():
    return lateral(ResNeXt50_32x4d_body_stride16)

class lateral(nn.Module):
    def __init__(self, conv_body_func):
        super().__init__()

        self.dim_in = [64, 256, 512, 1024, 2048]
        self.dim_in = self.dim_in[-1:0:-1]
        self.dim_out = [512, 256, 256, 256]

        self.num_lateral_stages = len(self.dim_in)
        self.topdown_lateral_modules = nn.ModuleList()

        for i in range(self.num_lateral_stages):
            self.topdown_lateral_modules.append(
                lateral_block(self.dim_in[i], self.dim_out[i]))

        self.bottomup = conv_body_func()
        dilation_rate = [2, 4, 6]
        encoder_stride = 16
        self.bottomup_top = ASPP_block(self.dim_in[0], self.dim_out[0], dilation_rate, encoder_stride)

    def _init_weights(self, init_type='xavier'):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
        def init_model_weight(m):
            for child_m in m.children():
                if not isinstance(child_m, nn.ModuleList):
                    child_m.apply(init_func)

        init_model_weight(self)

    def forward(self, x):
        _, _, h, w = x.shape
        backbone_stage_size = [(math.ceil(h/(2.0**i)), math.ceil(w/(2.0**i))) for i in range(5, 0, -1)]
        backbone_stage_size.append((h, w))
        bottemup_blocks_out = [self.bottomup.res1(x)]
        for i in range(1, self.bottomup.convX):
            bottemup_blocks_out.append(
                getattr(self.bottomup, 'res%d' % (i + 1))(bottemup_blocks_out[-1])
            )
        bottemup_top_out = self.bottomup_top(bottemup_blocks_out[-1])
        lateral_blocks_out = [bottemup_top_out]
        for i in range(self.num_lateral_stages):
            lateral_blocks_out.append(self.topdown_lateral_modules[i](
                bottemup_blocks_out[-(i + 1)]
            ))
        return lateral_blocks_out, backbone_stage_size


class ASPP_block(nn.Module):
    def __init__(self, dim_in, dim_out, dilate_rates, stride):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dilate_rates = dilate_rates
        self.aspp_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.aspp_conv3_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[0],
                                      dilation=self.dilate_rates[0], bias=False)
        self.aspp_conv3_2 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[1],
                                      dilation=self.dilate_rates[1], bias=False)
        self.aspp_conv3_3 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=self.dilate_rates[2],
                                      dilation=self.dilate_rates[2], bias=False)
        self.aspp_bn1x1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_2 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.aspp_bn3_3 = nn.BatchNorm2d(self.dim_out, momentum=0.5)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalpool_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.globalpool_bn = nn.BatchNorm2d(self.dim_out, momentum=0.5)

    def forward(self, x):
        x1 = self.aspp_conv1x1(x)
        x1 = self.aspp_bn1x1(x1)
        x2 = self.aspp_conv3_1(x)
        x2 = self.aspp_bn3_1(x2)
        x3 = self.aspp_conv3_2(x)
        x3 = self.aspp_bn3_2(x3)
        x4 = self.aspp_conv3_3(x)
        x4 = self.aspp_bn3_3(x4)

        x5 = self.globalpool(x)
        x5 = self.globalpool_conv1x1(x5)
        x5 = self.globalpool_bn(x5)
        w, h = x1.size(2), x1.size(3)
        x5 = nn.functional.interpolate(input=x5,size=(w, h),mode="bilinear",align_corners=True)

        out = torch.cat([x1, x2, x3, x4, x5], 1)
        return out


class lateral_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lateral = FTB_block(dim_in, dim_out)

    def forward(self, x):
        out = self.lateral(x)
        return out


class fcn_topdown(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim_in = [512, 256, 256, 256, 256, 256]
        self.dim_out = [256, 256, 256, 256, 256] + [1, ]

        self.num_fcn_topdown = len(self.dim_in)
        self.top_conv_num = 5
        self.top = nn.Sequential(
            nn.Conv2d(self.dim_in[0] * self.top_conv_num, self.dim_in[0], 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.dim_in[0], 0.5)
        )
        self.topdown_fcn1 = fcn_topdown_block(self.dim_in[0], self.dim_out[0])
        self.topdown_fcn2 = fcn_topdown_block(self.dim_in[1], self.dim_out[1])
        self.topdown_fcn3 = fcn_topdown_block(self.dim_in[2], self.dim_out[2])
        self.topdown_fcn4 = fcn_topdown_block(self.dim_in[3], self.dim_out[3])
        self.topdown_fcn5 = fcn_last_block(self.dim_in[4], self.dim_out[4])
        self.topdown_predict = fcn_topdown_predict(self.dim_in[5], self.dim_out[5])

        self.init_type = 'xavier'
        self._init_modules(self.init_type)

    def _init_modules(self, init_type):
        self._init_weights(init_type)

    def _init_weights(self, init_type='xavier'):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if init_type == 'kaiming':
                    nn.init.kaiming_normal(m.weight)
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.0)
                nn.init.constant_(m.bias.data, 0.0)

        for child_m in self.children():
            child_m.apply(init_func)

    def forward(self, laterals, backbone_stage_size):
        x = self.top(laterals[0])
        x1 = self.topdown_fcn1(laterals[1], x)
        x2 = self.topdown_fcn2(laterals[2], x1)
        x3 = self.topdown_fcn3(laterals[3], x2)
        x4 = self.topdown_fcn4(laterals[4], x3)
        x5 = self.topdown_fcn5(x4, backbone_stage_size)
        x6 = self.topdown_predict(x5)
        return x6


class fcn_topdown_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.afa_block = AFA_block(dim_in)
        self.ftb_block = FTB_block(self.dim_in, self.dim_out)

    def forward(self, lateral, top, size=None):
        if lateral.shape != top.shape:
            h, w = lateral.size(2), lateral.size(3)
            top = nn.functional.interpolate(input=top, size=(h, w), mode='bilinear',align_corners=True)
        out = self.afa_block(lateral, top)
        out = self.ftb_block(out)
        return out


class fcn_topdown_predict(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = nn.Dropout2d(0.0)
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x_softmax = self.softmax(x)
        return x, x_softmax


class FTB_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
        self.bn1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.relu(out)
        return out


class AFA_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_in = dim * 2
        self.dim_out = dim
        self.dim_mid = int(dim / 8)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.dim_mid, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.sigmd = nn.Sigmoid()

    def forward(self, lateral, top):
        w = torch.cat([lateral, top], 1)
        w = self.globalpool(w)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmd(w)
        out = w * lateral + top
        return out


class fcn_last_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ftb = FTB_block(dim_in, dim_out)

    def forward(self, input, backbone_stage_size):
        out = nn.functional.interpolate(input=input,size=(backbone_stage_size[4][0], backbone_stage_size[4][1]),mode="bilinear",align_corners=True)
        out = self.ftb(out)
        out = nn.functional.interpolate(input=out,size=(backbone_stage_size[5][0], backbone_stage_size[5][1]),mode="bilinear",align_corners=True)
        return out

