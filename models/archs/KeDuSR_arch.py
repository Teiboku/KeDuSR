import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ResList, PixelShufflePack, Res_Attention_Conf
from .matching import FeatureMatching, AlignedAttention
from .SISR import SISR_block
from .spynet import SpyNet, flow_warp
from .arch_util import FlowGuidedDCN
from .diffir.comman import CPEN,AdaIN



class FlowGuidedAlign(nn.Module):

    def __init__(self, nf=64, groups=8):
        super(FlowGuidedAlign, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_fea2X, ref_fea2X_flow, lr_shake_fea, flows):
        offset = torch.cat([ref_fea2X_flow, lr_shake_fea, flows], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        fea = self.lrelu(self.dcnpack(ref_fea2X, offset, flows))
        return fea

class KeDuSR(nn.Module):
    def __init__(self, args):

        self.args = args

        super(KeDuSR, self).__init__()
        n_feats = 64
        self.avgpool_2 = nn.AvgPool2d((2,2),(2,2))
        self.avgpool_4 = nn.AvgPool2d((4,4),(4,4))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        #SISR
        self.SISR = SISR_block()
        self.upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)

        #lr_nearby encoder
        self.lr_nearby_head = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1))
        self.lr_nearby_encoder = ResList(4, n_feats)
        self.upsample_lr_nearby = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)

        #ref encoder
        self.ref_head = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1))
        self.ref_encoder = ResList(4, n_feats)

        #SpyNet and Flow-guided DCN
        self.spynet = SpyNet('spynet_weight/spynet_sintel_final-3d2a1287.pth')
        self.spynet.requires_grad_(True)
        self.Ref_align = FlowGuidedAlign(nf=64, groups=8)

        #kernel-free matching  and warping
        self.feature_match = FeatureMatching(stride=1)
        self.ref_warp = AlignedAttention(scale=4, ksize=9)


        #alpha Likely of limited utility
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(1)
        self.alphaConf2X = nn.Sequential(nn.Conv2d(1, n_feats//4, 7, 1, 3),
                                         self.lrelu,
                                         nn.Conv2d(n_feats//4, n_feats, 3, 1, 1),
                                         self.lrelu)

        #AdaFuison
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

        self.cpen = CPEN() 
        self.adain_1 = AdaIN(n_feats*2, 512)
        self.adain_2 = AdaIN(n_feats*2, 512)
        self.adain_3 = AdaIN(n_feats, 512)


    def ada_and_tail(self, ref_fea_warped_h, confidence_map_up, lr_fea_up, cpen_embedding):
        cat_fea = torch.cat((ref_fea_warped_h * confidence_map_up, lr_fea_up), 1)            # 1 128 512 512
        
        cat_fea = self.adain_1(cat_fea, cpen_embedding)
        fusioned_fea = self.AdaFuison(cat_fea) # 1 128 512 512
        fusioned_fea = self.adain_2(fusioned_fea, cpen_embedding)

        fused_fea = self.fusion_tail(fusioned_fea) # 1 64 512 512
        fused_fea = self.adain_3(fused_fea, cpen_embedding)

        out = self.decoder(fused_fea)
        
        out = self.decoder_tail(out + lr_fea_up)
        return out
    
    def extract_features(self, lr, lr_nearby, ref):
        """提取基础特征和光流对齐"""
        # SISR特征提取
        lr_fea = self.SISR(lr)
        lr_fea_up = self.upsample(lr_fea)

        # 编码器特征提取
        lr_nearby_fea_up = self.lr_nearby_encoder(self.upsample_lr_nearby(self.lr_nearby_head(lr_nearby)))
        ref_fea = self.ref_encoder(self.ref_head(ref))

        # 光流估计和对齐
        ref_down = self.avgpool_2(ref)
        flows = self.spynet(lr_nearby, ref_down)
        flows_up = F.interpolate(input=flows, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        ref_fea_flow = flow_warp(ref_fea, flows_up.permute(0, 2, 3, 1))
        ref_fea_flow_align = self.Ref_align(ref_fea, ref_fea_flow, lr_nearby_fea_up, flows_up)
        
        return lr_fea_up, ref_fea_flow_align

    def match_and_warp(self, lr, lr_nearby, ref_fea_flow_align):
        """特征匹配和变形"""
        confidence_map, index_map = self.feature_match(lr, lr_nearby)
        ref_fea_warped = self.ref_warp(lr, index_map, ref_fea_flow_align)/5.0625

        # 高频特征提取
        ref_fea_warped_down = self.avgpool_4(ref_fea_warped)
        ref_fea_warped_h = 2 * self.alpha*(ref_fea_warped - F.interpolate(ref_fea_warped_down, scale_factor=4, mode='bicubic'))

        # 置信图上采样
        confidence_map_up = F.interpolate(confidence_map, scale_factor=4, mode='bicubic')
        confidence_map_up = self.alphaConf2X(confidence_map_up)
        
        return ref_fea_warped_h, confidence_map_up

    def get_cpen_embedding(self, lr, ref):
        cpen_embedding = self.cpen( lr, ref)
        return cpen_embedding

    def forward(self, lr, lr_nearby, ref, cpen_embedding=None):

        if self.training:
            # 1. 特征提取和对齐
            lr_fea_up, ref_fea_flow_align = self.extract_features(lr, lr_nearby, ref)
            # 2. 特征匹配和变形
            ref_fea_warped_h, confidence_map_up = self.match_and_warp(lr, lr_nearby, ref_fea_flow_align)
            # 3. 特征融合和重建
            out = self.ada_and_tail(ref_fea_warped_h, confidence_map_up, lr_fea_up, cpen_embedding)
            return out
        
        else:

            # 1. 特征提取和对齐
            lr_fea_up, ref_fea_flow_align = self.extract_features(lr, lr_nearby, ref)

            # 2. 分块处理
            p = self.args.chunk_size
            H, W = lr.shape[2]-16, lr.shape[3]-16
            num_x = W // p
            num_y = H // p

            sr_list = []
            for j in range(num_y):
                for i in range(num_x):
                    lr_patch = lr[:,:,j*(p):j*(p) + p+16, i*p:i*p + p+16]
                     
                    ref_fea_warped_h, confidence_map_up = self.match_and_warp(lr_patch, lr_nearby, ref_fea_flow_align)
                    
                    lr_fea_up_patch = lr_fea_up[:,:,j*(p*2):j*(p*2) + p*2+32, i*p*2:i*p*2 + p*2+32]                    
                    patch_sr = self.ada_and_tail(ref_fea_warped_h, confidence_map_up, lr_fea_up_patch, cpen_embedding)
                    sr_list.append(patch_sr[:,:,16:-16, 16:-16])
            
            # 3. 重组图像
            sr_list = torch.cat(sr_list, dim=0)
            sr_list = sr_list.view(sr_list.shape[0],-1)
            sr_list = sr_list.permute(1,0)
            sr_list = torch.unsqueeze(sr_list, 0)
            output = F.fold(sr_list, output_size=(H*2, W*2), kernel_size=(2*p,2*p), padding=0, stride=(2*p,2*p))
            
            return output




if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.chunk_size = 128

    args = Args()
    net = KeDuSR(args)
    image_size = 256
    image_tensor = torch.randn(1, 3, image_size*2, image_size*2)
    lr_nearby = torch.randn(1, 3, image_size, image_size)
    ref =  torch.randn(1, 3, image_size*2, image_size*2)
    output = net(image_tensor, lr_nearby, ref)  

    print(output.shape)