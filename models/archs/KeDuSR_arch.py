import torch
import torch.nn as nn
import torch.nn.functional as F

from .merge_vit import MergeVIT
from .common import ResList, PixelShufflePack, Res_Attention_Conf
from .SISR import SISR_block
from .diffir.comman import CPEN,AdaIN


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


        self.merge_vit = MergeVIT()


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


    def forward(self, lr, lr_nearby, ref, cpen_embedding=None):
        if self.training:
            lr_fea_up = self.extract_features(lr)
            lr_fea_vit = self.merge_vit(lr, ref)
            out = self.ada_and_tail(lr_fea_vit, lr_fea_up)
            return out
        else:
            lr_fea_up = self.extract_features(lr)
            p = self.args.chunk_size
            H, W = lr.shape[2]-16, lr.shape[3]-16
            num_x = W // p
            num_y = H // p
            sr_list = []
            ref_resized = F.interpolate(ref, size=(lr.shape[2]*2, lr.shape[3]), mode='bilinear', align_corners=False)

            for j in range(num_y):
                for i in range(num_x):
                    lr_patch = lr[:,:,j*(p):j*(p) + p+16, i*p:i*p + p+16]
                    ref_patch = ref_resized[:,:,j*(p*2):j*(p*2) + p*2+32, i*p*2:i*p*2 + p*2+32] 
                    lr_fea_up_patch = lr_fea_up[:,:,j*(p*2):j*(p*2) + p*2+32, i*p*2:i*p*2 + p*2+32]                                        
                    lr_fea_vit = self.merge_vit(lr_patch, ref_patch)
                    patch_sr = self.ada_and_tail(lr_fea_vit, lr_fea_up_patch, cpen_embedding)
                    sr_list.append(patch_sr[:,:,16:-16, 16:-16])
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
    with torch.no_grad():
        args = Args()
        net = KeDuSR(args).cuda()
        net.eval()
        
        cpen_embedding = torch.randn(1, 512).cuda()
        image_tensor = torch.randn(1, 3, 1296, 1808).cuda()  # LR
        lr_nearby = torch.randn(1, 3, 640, 896).cuda()       # LR_center
        ref = torch.randn(1, 3, 1280, 1792).cuda()           # Ref_SIFT
        output = net(image_tensor, lr_nearby, ref, cpen_embedding)

        print(output.shape)