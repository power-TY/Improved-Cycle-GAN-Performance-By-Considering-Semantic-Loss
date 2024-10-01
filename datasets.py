import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image



def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*"))


    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]) # img_B는 랜덤하게 샘플링

        # 만약 흑백(grayscale) 이미지라면 RGB 채널 이미지로 변환
        if img_A.mode != "RGB":
            img_A = to_rgb(img_A)
        if img_B.mode != "RGB":
            img_B = to_rgb(img_B)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))