import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
import glob
from tqdm import tqdm


class TrainSet(Dataset):
    def __init__(self, args):

        self.args = args

        LR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/LR', '*')))
        HR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/HR', '*')))
        Ref_full_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/Ref_full', '*')))
        self.scale = args.sr_scale

        self.LR_imgs = []
        self.HR_imgs = []
        self.Ref_full_imgs = []

        
        for i in tqdm(range(len(LR_list))):
            lr_img = cv2.imread(LR_list[i], -1)
            hr_img = cv2.imread(HR_list[i], -1)
            ref_img = cv2.imread(Ref_full_list[i], -1)
            if ref_img.shape[0] != lr_img.shape[0] * 2 or ref_img.shape[1] != lr_img.shape[1] * 2:
                print(f"索引 {i} 的图像尺寸不匹配：LR {lr_img.shape} vs Ref_SIFT {ref_img.shape}")
            self.LR_imgs.append(lr_img)
            self.HR_imgs.append(hr_img)
            self.Ref_full_imgs.append(ref_img)

    def __len__(self):
        return len(self.LR_imgs) * 50
    

    def crop_patch(self, LR, HR, Ref_full, p):
        # 获取 LR 与 Ref_full 的尺寸
        lr_h, lr_w = LR.shape[:2]
        ref_h, ref_w = Ref_full.shape[:2]
        lr_center_h, lr_center_w = ref_h // 2, ref_w // 2
        lr_center_start_y = (lr_h - lr_center_h) // 2
        lr_center_start_x = (lr_w - lr_center_w) // 2
        max_y = ref_h - p
        max_x = ref_w - p
        # 随机生成起始坐标
        rand_y = np.random.randint(0, max_y + 1)
        rand_x = np.random.randint(0, max_x + 1)
        # 在 Ref_full 上选择一个补丁（这里简单以 Ref_full 的中心为例）
        ref_patch_start_y = rand_y
        ref_patch_start_x = rand_x
        half_p = p//2
        ref_patch = Ref_full[ref_patch_start_y+half_p:ref_patch_start_y +p+half_p,
                             ref_patch_start_x+half_p:ref_patch_start_x +p+half_p, :]
        lr_patch_start_y = lr_center_start_y + (ref_patch_start_y // 2)
        lr_patch_start_x = lr_center_start_x + (ref_patch_start_x // 2)
        lr_patch = LR[lr_patch_start_y:lr_patch_start_y + p,
                      lr_patch_start_x:lr_patch_start_x + p, :]
        hr_patch = HR[lr_patch_start_y * 2 : (lr_patch_start_y + p) * 2,
              lr_patch_start_x * 2 : (lr_patch_start_x + p) * 2, :]
        #return lr  hr  lr_nearby  ref
        lr_patch_center = lr_patch[p//4 : p//4 + p//2,
                           p//4 : p//4 + p//2, :] 
        
        return lr_patch,hr_patch, lr_patch_center, ref_patch
    
    # todo 去掉rot90能不能涨点?
    def augment(self, *args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        k1 = np.random.randint(0, 3)
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]        
            
            img = np.rot90(img, k1)
            
            return img

        return [_augment(a) for a in args]


    def __getitem__(self, idx):

        idx = idx % len(self.LR_imgs)

        LR = self.LR_imgs[idx]
        HR = self.HR_imgs[idx]
        Ref_full = self.Ref_full_imgs[idx]


        lr, hr, lr_nearby, ref = self.crop_patch(LR, HR, Ref_full, p=self.args.patch_size)
        lr, hr, lr_nearby, ref = self.augment(lr, hr, lr_nearby, ref)

        sample = {
                    'lr': lr,
                    'lr_nearby': lr_nearby,
                    'ref': ref,
                    'hr': hr,
                }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key].copy()).permute(2, 0, 1).float()

        return sample



class TestSet_cache(Dataset):
    def __init__(self, args):
        LR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/LR', '*')))
        HR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/HR', '*')))
        LR_center_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/LR_center', '*')))
        Ref_SIFT_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/Ref_SIFT', '*')))
        self.scale = args.sr_scale
        self.LR_imgs = []
        self.HR_imgs = []
        self.LR_center_imgs = []
        self.Ref_SIFT_imgs = []
        self.names = []


        for i in range(len(LR_list)):
            self.names.append(os.path.basename(LR_list[i]))
            lr_img = cv2.imread(LR_list[i], -1)
            ref_sift_img = cv2.imread(Ref_SIFT_list[i], -1)
            if ref_sift_img.shape[0] != lr_img.shape[0] or ref_sift_img.shape[1] != lr_img.shape[1]:
                print(f"索引 {i} 的图像尺寸不匹配：LR {lr_img.shape} vs Ref_SIFT {ref_sift_img.shape}")
            self.LR_imgs.append(lr_img)
            self.Ref_SIFT_imgs.append(ref_sift_img)
            self.HR_imgs.append(cv2.imread(HR_list[i], -1))
            self.LR_center_imgs.append(cv2.imread(LR_center_list[i], -1))

    def __len__(self):
        return len(self.LR_imgs)


    def __getitem__(self, idx):

        sample = {'LR': self.LR_imgs[idx],
                 'HR': self.HR_imgs[idx],               
                  'LR_center': self.LR_center_imgs[idx],
                  'Ref_SIFT': self.Ref_SIFT_imgs[idx],
                  }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()
        sample['name'] = self.names[idx]
        return sample
