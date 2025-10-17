import cv2
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
import matplotlib.pyplot as plt
# from archs.mamba.mamba_ssm.modules.mamba_simple import Mamba
from model.mamba.mambapy.mamba import Mamba, MambaConfig
from torchvision import transforms

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):
        super(Fusion_Stem, self).__init__()


        self.stem11 = nn.Sequential(nn.Conv2d(16, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(32, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x3 = x
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_in_diff = torch.cat([x3-x1,x5-x3],2).view(N * D, 32, H, W)
        x_diff = self.stem12(x_in_diff)
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2
    
        return x
    

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.real

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  
        #     d_state=d_state,  
        #     d_conv=d_conv, 
        #     expand=expand  
        # )
        self.mamba_config = MambaConfig(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand_factor=expand,    # Block expansion factor
                n_layers=1
        )
        self.mamba = Mamba(self.mamba_config)
        
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba


class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
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

    def forward(self, x):
        B, D, C = x.size()
        #Multi-temporal Parallelization
        path = 3
        segment = 2**(path-1)
        tt = D // segment
        x_r = x.repeat(segment,1,1)
        x_o = x_r.clone()
        for i in range(1,segment):
            x_o[i*B:(i+1)*B,:D-i*tt,:] = x_r[i*B:(i+1)*B,i*tt:,:]
        x_o = self.attn(x_o)
        for i in range(1,segment):
            for j in range(i):
                x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] + x_o[B*(j+1):B*(j+2), tt*(i-j-1): tt*(i-j) , :]
            x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] / (i+1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


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
        #torch.floor(),返回一个新张量，包含输入input张量每个元素的floor，即取不大于元素的最大整数
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5

        #permute() 函数一次可以进行多个维度的交换
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w) ]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale2 * w), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)  # 以上为编码一个坐标
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

        mask = self.mask(x) # 一连串卷积

        # mask = self.up(mask)
        mask = F.interpolate(mask, (H, W), mode='bilinear', align_corners=False)
        mask = self.mask2(mask)
        mask = 1-mask

        adapted = self.adapt(x, scale, scale2)  # 增加放缩因子的一个卷积层

        return x + adapted * mask  # SFFM

        # #消融实验三
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

    def forward(self, x, scale, scale2):  # H和V方向放缩因子
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




class RhythmMamba_sda(nn.Module):
    def __init__(self, scale1,scale2,S=2, in_ch=3, 
                 depth=8, 
                 embed_dim=64, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.embed_dim = embed_dim

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([Block_mamba(
            dim = embed_dim, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        self.Mamba_upsample = nn.Upsample(scale_factor=(2,1,1))
        self.ConvBlockLast = nn.Conv3d(embed_dim, 1, kernel_size=1,stride=1, padding=0)

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.pooling = nn.AdaptiveAvgPool3d((None, S, S))
        
        self.scale1 = scale1
        self.scale2 = scale2

        self.S = S  # S is the spatial dimension of ST-rPPG block

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

        self.scale1 = torch.tensor(self.scale1)  # 标量
        self.scale2 = torch.tensor(self.scale2)  # 标量

        B, C, T, W, H = x.size()  # (B, 16, T, 128, 128)
        B2, C2, T2, W2, H2 = y.size()  # (B, 16, T, W2, H2)

        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, W, H)  # (B*T, 16, 128, 128)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(-1, C2, W2, H2)  # (B*T, 16, W2, H2)

        # 基准实验
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

        BT, C, W, H = x.size()  # (B*T, 16, 128, 128)
        x = x.view(B, T, C, W, H)  # (B, T, 16, 128, 128)
        y = y.view(B, T, C, W, H)  # (B, T, 16, 128, 128)

        x = torch.cat([x, y], dim=0).permute(0, 2, 1, 3, 4)  # (2*B, 16, T, 128, 128)

        # torch.Size([2, 16, 160, 128, 128])
        # print(f'x.shape : {x.shape}') 
        x = x.permute(0, 2, 1, 3, 4) # torch.Size([2, 160, 16, 128, 128])
        B, D, C, H, W = x.shape  # B=batch size, D=depth, C=channels, H=height, W=width
        # print(f'x.shape : {x.shape}') # torch.Size([2, 16, 160, 128, 128])
        # print(f'x.shape : {x.shape}') 
        x = self.Fusion_Stem(x)  # 输出: [N*D, C, H/8, W/8]
        # print(f'x.shape : {x.shape}')
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)  # 输出: [B, C, D, H/8, W/8]
        x = self.stem3(x)  # 输出: [B, embed_dim, D/2, H/8, W/8]

        mask = torch.sigmoid(x)  # 输出: [B, embed_dim, D/2, H/8, W/8]
        mask = self.attn_mask(mask)  # 输出: [B, embed_dim, D/2, H/8, W/8]
        x = x * mask  # 输出: [B, embed_dim, D/2, H/8, W/8]

        # x = torch.mean(x, 4)  # 输出: [B, embed_dim, D/2, H/8]
        # x = torch.mean(x, 3)  # 输出: [B, embed_dim, D/2]
        x =self.pooling(x)
        x = rearrange(x,'b c t h w ->b (t h w) c')
        
        for blk in self.blocks:
            x = blk(x)  # 输出: [B, D/2, embed_dim]

        x = rearrange(x, 'b (t h w) c -> b c t h w' ,h=2 ,w=2)  # 输出: [B, D/2, embed_dim]

        # rPPG = x.permute(0, 2, 1)  # 输出: [B, embed_dim, D/2]
        rPPG = self.Mamba_upsample(x)  # 输出: [B, embed_dim, D]
        x = self.ConvBlockLast(rPPG)    #[N, 1, D]
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (2*B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (2*B, 1, T)
        X = torch.cat(x_list + [x], 1)  # (2*B, M, T)
        return X

def load_model(args, device):
    return RhythmMamba_sda(0,0)

if __name__ == "__main__":
    model = RhythmMamba_sda(0,0)
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

