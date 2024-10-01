import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from models import GeneratorResNet
from datasets import ImageDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataName', type=str, default="monet2photo", help='data name')
    parser.add_argument('--size', type=int, default=64, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 데이터 로드
    transforms_ = transforms.Compose([
        transforms.Resize(int(opt.size), Image.BICUBIC), # 이미지 크기를 조금 키우기
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = ImageDataset(opt.dataName, transforms_=transforms_, mode='test')

    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # 생성자(generator)와 판별자(discriminator) 초기화 block수 이미지 사이즈에 따라 256이상이면 9로 설정
    G_AB = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)
    G_BA = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)

    G_AB.cuda()
    G_BA.cuda()

    # 생성자 파라미터 불러오기
    G_AB.load_state_dict(torch.load(opt.dataName+"_G_AB.pt"))
    G_BA.load_state_dict(torch.load(opt.dataName+"_G_BA.pt"))

    # test모드
    G_AB.eval()
    G_BA.eval()

    result_path='test_result_img/'+opt.dataName

    if not os.path.exists(result_path+'/A'):
        os.makedirs(result_path+'/A')
    if not os.path.exists(result_path+'/B'):
        os.makedirs(result_path+'/B')
    
    

    for i, batch in enumerate(test_dataloader):
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        save_image(fake_A, result_path+f"/A/%d.png" %(i+1), normalize=True)
        save_image(fake_B, result_path+f"/B/%d.png" %(i+1), normalize=True)





if __name__ == "__main__":
    main()