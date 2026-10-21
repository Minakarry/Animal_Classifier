"""数据集加载与预处理。"""
from __future__ import annotations

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from config import IMG_SIZE, MEAN, STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

normalize = transforms.Normalize(mean=MEAN, std=STD)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    def get_images(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        imgs_info = []
        for line in lines:
            # 兼容 tab / 多空格分隔
            if "\t" in line:
                path, label = line.split("\t", 1)
            else:
                path, label = line.rsplit(maxsplit=1)
            imgs_info.append((path, label))
        return imgs_info

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path).convert("RGB")
        img = self.train_tf(img) if self.train_flag else self.val_tf(img)
        return img, int(label)

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True,
    )
    for image, label in train_loader:
        print("image.shape = ", image.shape)
        print("label = ", label)
        break
