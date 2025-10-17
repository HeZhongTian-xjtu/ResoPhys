import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba.mambapy.mamba import Mamba, MambaConfig

from torch.nn import functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x*attention

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba_config = MambaConfig(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand_factor=expand,    # Block expansion factor
                n_layers=1
        )
        self.mamba = Mamba(self.mamba_config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out 

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


class PhysMamba(nn.Module):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMamba, self).__init__()

        self.Mamba_ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.Mamba_ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.Mamba_ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.Mamba_ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.Mamba_ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.Mamba_ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Mamba_Block1 = self._build_block(64, theta)
        self.Mamba_Block2 = self._build_block(64, theta)
        self.Mamba_Block3 = self._build_block(64, theta)
        # Fast Stream
        self.Mamba_Block4 = self._build_block(32, theta)
        self.Mamba_Block5 = self._build_block(32, theta)
        self.Mamba_Block6 = self._build_block(32, theta)

        # Upsampling
        self.Mamba_upsample1 = nn.Sequential(
            nn.Upsample(size=(frames//2,8,8)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.Mamba_upsample2 = nn.Sequential(
            nn.Upsample(size=(frames,8,8)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.Mamba_ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.Mamba_MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Mamba_MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.Mamba_fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.Mamba_fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.Mamba_drop_1 = nn.Dropout(drop_rate1)
        self.Mamba_drop_2 = nn.Dropout(drop_rate1)
        self.Mamba_drop_3 = nn.Dropout(drop_rate2)
        self.Mamba_drop_4 = nn.Dropout(drop_rate2)
        self.Mamba_drop_5 = nn.Dropout(drop_rate2)
        self.Mamba_drop_6 = nn.Dropout(drop_rate2)

        self.Mamba_poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, input): 
        x = input['input_clip']  # [B, C, T, H, W]
        x = torch.diff(x, n=1, dim=2, prepend=x[:, :, 0:1, :, :])  # [B, C, T, H, W]
        [batch, channel, length, width, height] = x.shape  # batch: B, channel: C, length: T, width: H, height: W

        x = self.Mamba_ConvBlock1(x)  # [B, 16, T, H, W]
        x = self.Mamba_MaxpoolSpa(x)  # [B, 16, T, H//2, W//2]
        x = self.Mamba_ConvBlock2(x)  # [B, 32, T, H//2, W//2]
        x = self.Mamba_ConvBlock3(x)  # [B, 64, T, H//2, W//2]
        x = self.Mamba_MaxpoolSpa(x)  # [B, 64, T, H//4, W//4]
    
        # Process streams
        s_x = self.Mamba_ConvBlock4(x)  # Slow stream: [B, 64, T//4, H//4, W//4]
        f_x = self.Mamba_ConvBlock5(x)  # Fast stream: [B, 32, T//4, H//4, W//4]

        # First set of blocks and fusion
        s_x1 = self.Mamba_Block1(s_x)  # [B, 64, T//4, H//4, W//4]
        s_x1 = self.Mamba_MaxpoolSpa(s_x1)  # [B, 64, T//4, H//8, W//8]
        s_x1 = self.Mamba_drop_1(s_x1)  # [B, 64, T//4, H//8, W//8]

        f_x1 = self.Mamba_Block4(f_x)  # [B, 32, T//4, H//4, W//4]
        f_x1 = self.Mamba_MaxpoolSpa(f_x1)  # [B, 32, T//4, H//8, W//8]
        f_x1 = self.Mamba_drop_2(f_x1)  # [B, 32, T//4, H//8, W//8]

        s_x1 = self.Mamba_fuse_1(s_x1, f_x1)  # [B, 64, T//4, H//8, W//8]

        # Second set of blocks and fusion
        s_x2 = self.Mamba_Block2(s_x1)  # [B, 64, T//4, H//8, W//8]
        s_x2 = self.Mamba_MaxpoolSpa(s_x2)  # [B, 64, T//4, H//16, W//16]
        s_x2 = self.Mamba_drop_3(s_x2)  # [B, 64, T//4, H//16, W//16]
        
        f_x2 = self.Mamba_Block5(f_x1)  # [B, 32, T//4, H//8, W//8]
        f_x2 = self.Mamba_MaxpoolSpa(f_x2)  # [B, 32, T//4, H//16, W//16]
        f_x2 = self.Mamba_drop_4(f_x2)  # [B, 32, T//4, H//16, W//16]

        s_x2 = self.Mamba_fuse_2(s_x2, f_x2)  # [B, 64, T//4, H//16, W//16]
        
        # Third blocks and upsampling
        s_x3 = self.Mamba_Block3(s_x2)  # [B, 64, T//4, H//16, W//16]
        s_x3 = self.Mamba_upsample1(s_x3)  # [B, 64, T//2, H//8, W//8]
        s_x3 = self.Mamba_drop_5(s_x3)  # [B, 64, T//2, H//8, W//8]

        f_x3 = self.Mamba_Block6(f_x2)  # [B, 32, T//4, H//16, W//16]
        f_x3 = self.Mamba_ConvBlock6(f_x3)  # [B, 32, T//4, H//16, W//16]
        f_x3 = self.Mamba_drop_6(f_x3)  # [B, 32, T//4, H//16, W//16]

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1)  # [B, 96, T//2, H//8, W//8]
        x_final = self.Mamba_upsample2(x_fusion)  # [B, 48, T, H, W]

        x_final = self.Mamba_poolspa(x_final)  # [B, 48, T, 1, 1]
        x_final = self.Mamba_ConvBlockLast(x_final)  # [B, 1, T, 1, 1]

        rPPG = x_final.view(-1, length)  # [B, T]

        return {
            'rPPG': rPPG,  # [B, T]
        } 


if __name__ == '__main__':
    #! cal params. and MACs
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

    model = PhysMamba(theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=160).cuda()
    model.eval()
    input_data = {'input_clip' : torch.randn(1, 3, 160, 128, 128).cuda(), 'epoch' : 0}
    import time
    start_time = time.time()
    for i in range(100):
        output = model(input_data)['rPPG']
    end_time = time.time()
    print(f'cost time: {(end_time-start_time)/160*10} ms')
    # output = model(input_data)['rPPG']
    # print(f'output.shape: {output.shape}')
    # prof = FlopsProfiler(model)
    # prof.start_profile(ignore_list=[type(nn.Upsample())])
    # output = model(input_data)['rPPG']
    # params = prof.get_total_params(as_string=True)
    # flops = prof.get_total_macs(as_string=True)
    # print(f'MACs: {flops}, Params: {params}')
    # print(f'output.shape: {output.shape}')
    # prof.end_profile()
