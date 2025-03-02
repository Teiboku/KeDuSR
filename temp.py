import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        base = self.upsample(x)
        res = self.res_block(x)
        return self.final_process(base + res)

class MergeVIT(nn.Module):
    def __init__(self, model_name='nextvit_small.bd_in1k', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained, 
            features_only=True,
        )
        
        config = timm.data.resolve_model_data_config(self.backbone)
        mean = torch.tensor(config['mean']).view(1, 3, 1, 1)
        std = torch.tensor(config['std']).view(1, 3, 1, 1)
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.upsampler = ResidualUpsampler(in_channels=96)

    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def cut_tensor(self, x):
        return x[:, :, :x.shape[-1], :]

            
    def forward(self, lr,ref):
        x = self.surround_with_refs(lr, ref)
        x = self.normalize(x)
        features = self.backbone(x)
        features = [self.cut_tensor(feat) for feat in features]
        feat = features[0]

        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
    
        return   self.upsampler(feat)
    
    def merge_lr_and_ref(self, lr, ref):
        b, c, h, w = lr.shape
        refs_combined = torch.cat([ref, ref], dim=3)
        return torch.cat([lr, refs_combined], dim=2)

    def surround_with_refs(self, lr, ref):
        """
        用12个ref图像围绕lr形成一个新的正方形
        
        布局如下 (每个数字代表一个ref的位置):
        1  2  3  4
        5  LR LR  6
        7  LR LR  8
        9  10 11 12
        
        参数:
            lr: 形状为 (b, c, h, w) 的张量
            ref: 形状为 (b, c, h/2, w/2) 的张量
            
        返回:
            square: 形状为 (b, c, 2h, 2w) 的正方形张量
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

if __name__ == "__main__":
    model = MergeVIT()
    model.eval()
    
    lr = torch.randn(1, 3, 256, 256)
    ref = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        features = model(lr,ref)
        print(features.shape)
