import timm
import torch
import torch.nn as nn
from torchvision import transforms

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
        
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def cut_tensor(self, x):
        return x[:, :, :x.shape[-1], :]

            
    def forward(self, lr,ref):
        x = self.merge_lr_and_ref(lr, ref)
        x = self.normalize(x)
        features = self.backbone(x)
        features = [self.cut_tensor(feat) for feat in features]
        return features[0]
    
    def merge_lr_and_ref(self, lr, ref):
        b, c, h, w = lr.shape
        refs_combined = torch.cat([ref, ref], dim=3)
        return torch.cat([lr, refs_combined], dim=2)


if __name__ == "__main__":
    model = MergeVIT()
    model.eval()
    
    lr = torch.randn(1, 3, 256, 256)
    ref = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        features = model(lr,ref)
        print(features.shape)
