import torch
from torchvision import datasets, transforms
import math
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch.nn as nn


def main():

    # A 폴더와 B 폴더의 이미지 데이터셋 경로 설정
    data_root_A = 'FID_score/origin/'
    data_root_B = 'FID_score/VGG/'

    # 데이터 변환 설정
    data_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 이미지 픽셀 값을 [-1, 1]로 정규화
    ])

    # 데이터셋 로드
    dataset_A = datasets.ImageFolder(data_root_A, transform=data_transform)
    dataset_B = datasets.ImageFolder(data_root_B, transform=data_transform)

    # 데이터로더 설정
    batch_size = 1
    dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=batch_size, shuffle=False, num_workers=4)

    
    # Inception V3 모델을 불러옵니다.
    inception_model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
    inception_model.eval()
    inception_model.fc = nn.Identity()
    inception_model.cuda()


    def extract_features(dataloader, model):
        all_features = []
        for batch in dataloader:
            images, _ = batch
            if torch.cuda.is_available():
                images = images.cuda()
            with torch.no_grad():
                features = model(images)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    # compute embeddings for real images
    real_image_embeddings = extract_features(dataloader_A, inception_model)

    # compute embeddings for generated images
    generated_image_embeddings = extract_features(dataloader_B, inception_model)

    real_image_embeddings.shape, generated_image_embeddings.shape

    # FID 스코어 계산
    from scipy.linalg import sqrtm

    mu1, sigma1 = np.mean(real_image_embeddings, axis=0), np.cov(real_image_embeddings, rowvar=False)
    mu2, sigma2 = np.mean(generated_image_embeddings, axis=0), np.cov(generated_image_embeddings, rowvar=False)

    # Covariance 루트를 계산합니다.
    ssqrt = sqrtm(sigma1 @ sigma2)

    # 두 데이터셋 간의 Fréchet Distance를 계산합니다.
    fid_score = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * ssqrt)

    print(f"FID Score: {fid_score:.2f}")







if __name__ == "__main__":
    main()
