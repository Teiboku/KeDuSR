import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
attention_maps = []
def hook_attention(module, input, output):
    """ 提取 Attention Map """
    q, k, v = module.q(input[0]), module.k(input[0]), module.v(input[0])  # 获取 q, k, v
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])  # 计算 Attention
    attn_weights = F.softmax(attn_weights, dim=-1)  # 归一化
    attention_maps.append(attn_weights)  



class ResidualUpsampler(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        # 基础上采样路径
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        # 残差学习路径
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(scale_factor)
        )

        self.final_process = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        base = self.upsample(x)
        res = self.res_block(x)
        return self.final_process(base + res)

class MergeVIT(nn.Module):
    def __init__(self, model_name='nextvit_small.bd_in1k', pretrained=True, debug = False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained, 
            features_only=True,
        )
        if debug:
            for name, module in self.backbone.named_modules():
                if "e_mhsa" in name and not any(x in name for x in [".q", ".k", ".v", ".proj", ".attn_drop", ".proj_drop", ".sr", ".norm"]):
                    print(f"Registering hook on: {name}")
                    module.register_forward_hook(hook_attention)


        config = timm.data.resolve_model_data_config(self.backbone)
        mean = torch.tensor(config['mean']).view(1, 3, 1, 1)
        std = torch.tensor(config['std']).view(1, 3, 1, 1)
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.conv = nn.Conv2d(96, 64, kernel_size=3, padding=1)



    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def cut_tensor(self, x):
        return x[:, :, :x.shape[-1], :]

            
    def forward(self, lr,ref):
        # Upsample lr to match ref size
        lr = F.interpolate(lr, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.surround_with_refs(lr, ref)
        x = self.normalize(x)

        features = self.backbone(x)
        features = [self.cut_tensor(feat) for feat in features]
        feat = features[0]
        return self.conv(feat)
    
    def merge_lr_and_ref(self, lr, ref):
        b, c, h, w = lr.shape
        refs_combined = torch.cat([ref, ref], dim=3)
        return torch.cat([lr, refs_combined], dim=2)

    def surround_with_refs(self, lr, ref):
        """
        """
        b, c, h, w = lr.shape
        _, _, ref_h, ref_w = ref.shape
        
        # 确保ref的尺寸是lr的一半
        assert ref_h == h//2 and ref_w == w//2, f"ref应为lr的一半大小，lr:{(h,w)}，ref:{(ref_h,ref_w)}"
        
        # 创建一个2h x 2w的空白画布
        result = torch.zeros(b, c, h*2, w*2, device=lr.device)
        
        # 放置lr在中心
        result[:, :, h//2:h//2+h, w//2:w//2+w] = lr
        
        # 放置12个ref围绕lr
        # 上方一行 (4个ref)
        result[:, :, 0:h//2, 0:w//2] = ref  # 位置1
        result[:, :, 0:h//2, w//2:w] = ref  # 位置2
        result[:, :, 0:h//2, w:w+w//2] = ref  # 位置3
        result[:, :, 0:h//2, w+w//2:2*w] = ref  # 位置4
        
        # 左右两侧 (4个ref)
        result[:, :, h//2:h, 0:w//2] = ref  # 位置5
        result[:, :, h//2:h, w+w//2:2*w] = ref  # 位置6
        result[:, :, h:h+h//2, 0:w//2] = ref  # 位置7
        result[:, :, h:h+h//2, w+w//2:2*w] = ref  # 位置8
        
        # 下方一行 (4个ref)
        result[:, :, h+h//2:2*h, 0:w//2] = ref  # 位置9
        result[:, :, h+h//2:2*h, w//2:w] = ref  # 位置10
        result[:, :, h+h//2:2*h, w:w+w//2] = ref  # 位置11
        result[:, :, h+h//2:2*h, w+w//2:2*w] = ref  # 位置12
        return result


import torchvision.utils as vutils
from PIL import Image

def save_attention_map(attn_maps, layer_idx=0, save_path="attention_map.png"):
    vutils.save_image(attn_maps[layer_idx], save_path)
# 调用保存方法
if __name__ == "__main__":
    model = MergeVIT().cuda()
    model.eval()


    lr_path = '/root/autodl-tmp/reproduce/KeDuSR/dataset/CameraFusion-Real/test/LR/IMG_8.png'
    ref_path = '/root/autodl-tmp/reproduce/KeDuSR/dataset/CameraFusion-Real/test/Ref/IMG_8.png'
    lr = Image.open(lr_path)
    ref = Image.open(ref_path)
    # Crop 256x256 patch from center
    lr = torch.from_numpy(np.array(lr)).permute(2, 0, 1).unsqueeze(0).float().cuda()
    ref = torch.from_numpy(np.array(ref)).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    _, _, h, w = lr.shape
    top = (h - 256) // 2
    left = (w - 256) // 2

    lr = lr[:, :, top:top+256, left:left+256]
    ref = ref[:, :, top:top+256, left:left+256]

    with torch.no_grad():
        features = model(lr,ref)
        print(features.shape)
    print(f"提取到 {len(attention_maps)} 层 Attention Map")
    print(f"attention 0 max:{attention_maps[0].max()}, min:{attention_maps[0].min()}")
    print(f"attention 1 max:{attention_maps[1].max()}, min:{attention_maps[1].min()}")
    print(f"attention 2 max:{attention_maps[2].max()}, min:{attention_maps[2].min()}")
    print(f"attention 3 max:{attention_maps[3].max()}, min:{attention_maps[3].min()}")
    

    save_attention_map(attention_maps*255, layer_idx=0, save_path="layer0_attention.png")
    save_attention_map(attention_maps*255, layer_idx=1, save_path="layer1_attention.png")
    save_attention_map(attention_maps*255, layer_idx=2, save_path="layer2_attention.png")
    save_attention_map(attention_maps*255, layer_idx=3, save_path="layer3_attention.png")
