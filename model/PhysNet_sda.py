import cv2
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from torchvision import transforms




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()
        scale = scale.item()
        scale2 = scale2.item()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        # torch.floor() returns a new tensor containing the floor of each element
        # (largest integer not greater than the element)
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5

        # permute() swaps multiple dimensions at once
        coor_h = coor_h.permute(1, 0)

        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w) ]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale2 * w), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)  # encode the coordinate
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.out_h = 1
        self.out_w = 1
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2,ceil_mode=True),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.Upsample((self.out_h,self.out_w), mode='bilinear', align_corners=False),
            # nn.Conv2d(16, 1, 3, 1, padding=1),
            # nn.BatchNorm2d(1),
            # nn.Sigmoid()
        )

        self.mask2 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)
        # self.up = nn.Upsample((H, W), mode='bilinear', align_corners=False)


    def forward(self, x, scale, scale2):
        B,T,H,W = x.size()
        mask = self.mask(x)  # a sequence of convolutional layers

        # upsample mask to input spatial size
        mask = F.interpolate(mask, (H, W), mode='bilinear', align_corners=False)
        mask = self.mask2(mask)
        mask = 1 - mask

        adapted = self.adapt(x, scale, scale2)  # convolutional layer that accounts for scaling factors

        return x + adapted * mask  # SFFM

    # # Ablation experiment 3
        # B,T,H,W = x.size()
        # adapted = self.adapt(x, scale, scale2)
        # return adapted

    def set_w_h(self,w,h):
        self.out_w = w
        self.out_h = h


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):  # horizontal and vertical scaling factors
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        x = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return x


def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = x + res
        return res

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat,kernel_size))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = x + res
        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




class PhysNet_sda(nn.Module):
    def __init__(self, scale1,scale2,S=2, in_ch=3):
        super().__init__()

        self.scale1 = scale1
        self.scale2 = scale2

        self.S = S  # S is the spatial dimension of ST-rPPG block

        # self.start = nn.Sequential(
        #     nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
        #     nn.BatchNorm3d(32),
        #     nn.ELU()
        # )

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_ch, 16, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock1_1 = nn.Sequential(
            nn.Conv3d(in_ch, 16, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        # self.ConvBlock2 = nn.Sequential(
        #     nn.Conv3d(16, 32, [1, 5, 5], stride=1, padding=[0, 2, 2]),
        #     nn.BatchNorm3d(32),
        #     nn.ELU(),
        # )

        # self.ConvBlock2_1 = nn.Sequential(
        #     nn.Conv3d(16, 32, [1, 5, 5], stride=1, padding=[0, 2, 2]),
        #     nn.BatchNorm3d(32),
        #     nn.ELU(),
        # )

        # self.ConvBlock3 = nn.Sequential(
        #     nn.Conv3d(32, 64, [1, 5, 5], stride=1, padding=[0, 2, 2]),
        #     nn.BatchNorm3d(64),
        #     nn.ELU(),
        # )
        #
        # self.ConvBlock3_1 = nn.Sequential(
        #     nn.Conv3d(32, 64, [1, 5, 5], stride=1, padding=[0, 2, 2]),
        #     nn.BatchNorm3d(64),
        #     nn.ELU(),
        # )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )



        self.sa_adapt_1 = SA_adapt(16)  # SFFM

        # scale-aware upsampling layer
        self.sa_upsample_1 = SA_upsample(16)

        self.conv2d_1 = nn.Conv2d(16, 16, 3, padding=1, bias=True)

        self.sa_adapt_2 = SA_adapt(16)

        # scale-aware upsampling layer
        self.sa_upsample_2 = SA_upsample(16)

        self.conv2d_2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)

        self.conv2d_3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)

        self.conv2d_4 = nn.Conv2d(16, 16, 3, padding=1, bias=True)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        self.upsample = nn.Upsample((128, 128), mode='bicubic', align_corners=False)

    def set_scale(self, scale1,scale2):
        self.scale1 = scale1
        self.scale2 = scale2

    def forward(self, x, y):
        B, C, T, H, W = x.size()  # (B, C, T, H, W)

        parity = []
        x = self.ConvBlock1(x)  # (B, 16, T, 128, 128)
        w = x.size(3)
        self.scale1 = 128.0 / w
        # x = self.ConvBlock2(x)
        # x = self.ConvBlock3(x)
        y = self.ConvBlock1_1(y)  # (B, 16, T, W2, H2)
        w = y.size(3)
        self.scale2 = 128.0 / w
        # y = self.ConvBlock2_1(y)
        # y = self.ConvBlock3_1(y)

        # x = x.permute((0,2,1,3,4))# (B, T, C, 128, 128)
        # y = y.permute((0, 2, 1, 3, 4))# (B, T, C, 128, 128)

        self.scale1 = torch.tensor(self.scale1)  # scalar
        self.scale2 = torch.tensor(self.scale2)  # scalar

        B, C, T, W, H = x.size()  # (B, 16, T, 128, 128)
        B2, C2, T2, W2, H2 = y.size()  # (B, 16, T, W2, H2)

        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, W, H)  # (B*T, 16, 128, 128)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(-1, C2, W2, H2)  # (B*T, 16, W2, H2)

        # baseline experiment
        res = x  # (B*T, 16, 128, 128)
        res2 = y  # (B*T, 16, W2, H2)
        res = self.sa_adapt_1(res, self.scale1, self.scale1)  # (B*T, 16, 128, 128)
        res2 = self.sa_adapt_2(res2, self.scale2, self.scale2)  # (B*T, 16, W2, H2)
        res = self.conv2d_1(res)  # (B*T, 16, 128, 128)
        res2 = self.conv2d_2(res2)  # (B*T, 16, W2, H2)
        res = x + res  # (B*T, 16, 128, 128)
        res2 = y + res2  # (B*T, 16, W2, H2)
        res = self.sa_upsample_1(res, self.scale1, self.scale1)  # (B*T, 16, 128, 128)
        res2 = self.sa_upsample_2(res2, self.scale2, self.scale2)  # (B*T, 16, 128, 128)
        x = self.conv2d_3(res)  # (B*T, 16, 128, 128)
        y = self.conv2d_4(res2)  # (B*T, 16, 128, 128)

        # # Ablation experiment 1
        # x = self.upsample(x)
        # y = self.upsample(y)

        # Ablation experiment 2
        # x = self.sa_upsample_1(x,self.scale1, self.scale1)
        # y = self.sa_upsample_2(y, self.scale2, self.scale2)
        # x = self.conv2d_3(x)
        # y = self.conv2d_4(y)

        # # Ablation experiment 3
        # res = x
        # res2 = y
        # res = self.sa_adapt_1(res, self.scale1, self.scale1)
        # res2 = self.sa_adapt_2(res2, self.scale2, self.scale2)
        # res = self.conv2d_1(res)
        # res2 = self.conv2d_2(res2)
        # res = x + res
        # res2 = y + res2
        # res = self.sa_upsample_1(res, self.scale1, self.scale1)
        # res2 = self.sa_upsample_2(res2, self.scale2, self.scale2)
        # x = self.conv2d_3(res)
        # y = self.conv2d_4(res2)

        BT, C, W, H = x.size()  # (B*T, 16, 128, 128)
        x = x.view(B, T, C, W, H)  # (B, T, 16, 128, 128)
        y = y.view(B, T, C, W, H)  # (B, T, 16, 128, 128)

        x = torch.cat([x, y], dim=0).permute(0, 2, 1, 3, 4)  # (2*B, 16, T, 128, 128)

        x = self.loop1(x)  # (2*B, 64, T, 64, 64)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)  # (2*B, 64, T/2, 32, 32)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)  # (2*B, 64, T/4, 16, 16)
        x = self.loop4(x)  # (2*B, 64, T/4, 8, 8)

        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (2*B, 64, T/2, 8, 8)
        x = self.decoder1(x)  # (2*B, 64, T/2, 8, 8)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')  # (2*B, 64, T/2+1, 8, 8) if parity[-1] == 1
        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (2*B, 64, T, 8, 8)
        x = self.decoder2(x)  # (2*B, 64, T, 8, 8)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')  # (2*B, 64, T+1, 8, 8) if parity[-2] == 1
        x = self.end(x)  # (2*B, 1, T, S, S)

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (2*B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (2*B, 1, T)
        X = torch.cat(x_list + [x], 1)  # (2*B, M, T)
        return X

def load_model(args, device):
    return PhysNet_sda(0,0)

if __name__ == "__main__":
    model = PhysNet_sda(0,0)
    model.cuda()

    import time

    import torchkeras
    input_data = torch.rand(1, 3, 160, 85, 85).cuda()
    start_time = time.time()
    with torch.no_grad():  # 确保在推理期间不计算梯度
        for i in range(100):
            y = model(input_data, input_data)
    end_time = time.time()

    inference_time = (end_time - start_time) / 100
    print("160 frames inference time: %.4f s" % inference_time)
    print("30s inference time: %.4f s" % (30 * 30 /160 * inference_time))
    print("inference fps: %.4f fps" % (1 /((inference_time)/160)))
    print("inference time per frame: %.4f ms" % ((inference_time)/160 * 1000))
    # print(y["rppg"].shape)
    torchkeras.summary(model, input_data_args=(input_data, input_data))

    from thop import profile
    flops, params = profile(model, (input_data, input_data))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

