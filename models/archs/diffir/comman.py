import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

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


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class CPEN(nn.Module):
    def __init__(self,n_feats = 128, n_encoder_res = 6):
        super(CPEN, self).__init__()
        E1=[nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)

        fea = self.E(x).squeeze(-1).squeeze(-1)

        fea1 = self.mlp(fea)
        return fea1



import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """
    AdaIN: Adaptive Instance Normalization
    假设输入特征: (N, C, H, W)
         条件向量: (N, style_dim)
    """
    def __init__(self, num_features, style_dim):
        """
        :param num_features: 特征图通道数 C
        :param style_dim: 条件/风格向量的维度
        """
        super(AdaIN, self).__init__()
        # 用于将一维向量映射为对通道进行缩放(γ)和偏移(β)的系数
        self.fc = nn.Linear(style_dim, num_features * 2)

        # InstanceNorm 不学习 γ, β（设 affine=False）
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, style_code):
        """
        :param x: shape [N, C, H, W]
        :param style_code: shape [N, style_dim]
        :return: 与 x 同形状的张量
        """
        # 1) 先对特征 x 做 Instance Normalization
        normalized = self.instance_norm(x)  # [N, C, H, W]

        # 2) style_code 映射为 2*C 大小的向量, 每一半分别对应 γ 和 β
        #    形状 [N, 2*C]
        style_params = self.fc(style_code)  
        # 分割得到 γ, β
        # 形状 [N, C]
        gamma, beta = style_params.chunk(2, dim=1)

        # 3) 需要将 gamma, beta 扩展到与特征图相同的 (N, C, H, W)
        #    以便在通道维度进行广播
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [N, C, 1, 1]

        # 4) 自适应变换: out = normalized * gamma + beta
        out = normalized * gamma + beta
        return out
    




if __name__ == "__main__":
    image_size = 128
    image_tensor = torch.randn(1, 3, image_size, image_size)
    cpen = CPEN()
    one_dim_tensor = cpen(image_tensor, image_tensor)
    print(one_dim_tensor.shape)
    adain = AdaIN(128, 512)
    adain_in = torch.randn(1, 128, image_size, image_size)
    adain_out = adain(adain_in, one_dim_tensor)
    print(adain_out.shape)