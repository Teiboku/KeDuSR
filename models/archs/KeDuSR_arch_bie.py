
import torch
import torch.nn as nn
import torch.nn.functional as F




from .common import ResList, PixelShufflePack, Res_Attention_Conf
from .SISR import SISR_block
import math


 
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        #initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class BIE(nn.Module):
    def __init__(self, nf=64):
        super(BIE, self).__init__()
        # self-process
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1

        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        # initialization
        #initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2, x_s):
        b, c, h, w = x_1.shape

        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s

        return out_1 + x_2_, out_2 + x_1_, x_s_


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

    

def process_image_patches(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor, patch_size: int = 256, overlap: int = 16):
    """
    Process two input tensors by splitting them into patches, running through model and combining results
    
    Args:
        model: Neural network model that takes two BCHW tensors and outputs one BC(2H)(2W) tensor
        x1: First input tensor of shape (B,C,H,W) 
        x2: Second input tensor of shape (B,C,H,W)
        patch_size: Size of patches to process
        overlap: Overlap between patches to avoid boundary artifacts
        
    Returns:
        Combined output tensor of shape (B,C,2H,2W)
    """
    assert x1.shape == x2.shape, "Input tensors must have same shape"
    B, C, H, W = x1.shape
    
    # Calculate number of patches
    stride = patch_size - overlap
    n_patches_h = math.ceil((H - patch_size) / stride) + 1
    n_patches_w = math.ceil((W - patch_size) / stride) + 1
    
    # Initialize output tensor
    out_h, out_w = 2*H, 2*W
    output = torch.zeros((B, C, out_h, out_w), device=x1.device)
    count = torch.zeros((B, C, out_h, out_w), device=x1.device)
    
    # Process each patch
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch coordinates
            h_start = i * stride
            w_start = j * stride
            h_end = min(h_start + patch_size, H)
            w_end = min(w_start + patch_size, W)
            
            # Extract patches
            patch1 = x1[:, :, h_start:h_end, w_start:w_end]
            patch2 = x2[:, :, h_start:h_end, w_start:w_end]
            
            # Process patches
            with torch.no_grad():
                out_patch = model(patch1,patch1, patch2)
            
            # Calculate output coordinates
            out_h_start = h_start * 2
            out_w_start = w_start * 2
            out_h_end = h_end * 2
            out_w_end = w_end * 2
            
            # Add to output with overlap handling
            output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += out_patch
            count[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1
    
    # Average overlapping regions
    output = output / (count + 1e-8)
    
    return output


class KeDuSR(nn.Module):
    def __init__(self, args):

        self.args = args

        super(KeDuSR, self).__init__()
        n_feats = 64
        self.avgpool_2 = nn.AvgPool2d((2,2),(2,2))
        self.avgpool_4 = nn.AvgPool2d((4,4),(4,4))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.SISR = SISR_block()

        self.upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)
        
 
        self.SISR_ref = SISR_block()
    
        self.SISR_lr  = SISR_block()
        self.lr_upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)



        self.AdaFuison = Res_Attention_Conf(n_feats*2, n_feats*2, res_scale=1, SA=True, CA=True)
        self.fusion_tail = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 3, 1, 1),
            self.lrelu
        )


        #decoder
        self.decoder = ResList(4, n_feats)
        self.decoder_tail = nn.Sequential(nn.Conv2d(n_feats, n_feats//2, 3, 1, 1),
                                         self.lrelu,
                                         nn.Conv2d(n_feats//2, 3, 3, 1, 1))

        self.bie = BIE()
        self.reduce_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)



    def ada_and_tail(self, lr_fea_vit, lr_fea_up):
        cat_fea = torch.cat((lr_fea_vit, lr_fea_up), 1)            # 1 96+64= 160 512 512
        fusioned_fea = self.AdaFuison(cat_fea) # 1 128 512 512
        fused_fea = self.fusion_tail(fusioned_fea) # 1 64 512 512
        out = self.decoder(fused_fea)
    
        out = self.decoder_tail(out + lr_fea_up)
        return out
    
    def extract_features(self, lr):
        lr_fea = self.SISR(lr)
        lr_fea_up = self.upsample(lr_fea)        
        return lr_fea_up
    
    def extract_lr_ref_feat(self, lr, ref):
        lr_fea = self.SISR_lr(lr)  # 提取 lr 的特征
        lr_fea_up = self.lr_upsample(lr_fea)  # 上采样得到更大的特征图
        ref = self.SISR_ref(ref)  # 提取 ref 的特征

        # 获取形状
        B, C, H, W = lr_fea_up.shape  # 大 tensor
        _, _, h, w = ref.shape  # 小 tensor
        # 计算小 tensor 的插入位置
        start_H, start_W = (H - h) // 2, (W - w) // 2
        end_H, end_W = start_H + h, start_W + w
        ref_up = lr_fea_up
        ref_up[:, :, start_H:end_H, start_W:end_W] = ref
        return lr_fea_up, ref_up
    

    def forward(self, lr, lr_nearby, ref):
        lr_fea_up = self.extract_features(lr)
        lr_in, ref_in = self.extract_lr_ref_feat(lr,ref)
        o1,o2,o3 = self.bie(lr_in,ref_in,lr_fea_up)
        in_feat = torch.concat((o1,o2),dim=1)
        in_feat = self.reduce_conv(in_feat)
        out = self.ada_and_tail(in_feat, o3)
        
        return out
