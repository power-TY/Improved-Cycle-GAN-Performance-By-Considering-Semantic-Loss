import glob


import argparse
import os
import gc

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image

# 모델 구조 바꾼거 테스트 시 다른걸로 바꾸기
# from models import GeneratorResNet
from models_LeakyReLU import GeneratorResNet
# from models_VGG import GeneratorResNet





def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms_
        self.files_B = sorted(glob.glob(os.path.join(root) + "/*"))

    def __getitem__(self, index):
        img_B = Image.open(self.files_B[index % len(self.files_B)])
      
        # 만약 흑백(grayscale) 이미지라면 RGB 채널 이미지로 변환
        if img_B.mode != "RGB":
            img_B = to_rgb(img_B)

        img_B = self.transform(img_B)
        return {"B": img_B}
    
    def __len__(self):
        return (len(self.files_B))
    
    

def main():

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    # 데이터 로드
    transforms_ = transforms.Compose([
        transforms.Resize((int(opt.size),int(opt.size)),  Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # weight 파라미터 경로
    weight_path='weight/torch/'+opt.dataName+'_Hybrid'+'/'

    #불러올 epoch 설정 디폴트는 200
    epoch=200

    test_dataset = ImageDataset(opt.dataName+"/test/B", transforms_=transforms_)

    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    # 생성자(generator)와 판별자(discriminator) 초기화 block수 이미지 사이즈에 따라 256이상이면 9로 설정
    G_BA = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)
    # G_AB = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)

    # cuda 사용시 이용 대용량 이미지 쉽지않음.
    torch.cuda.empty_cache()
    gc.collect()
    G_BA.cuda()
    # G_AB.cuda()
    # 생성자 파라미터 불러오기
    G_BA.load_state_dict(torch.load(weight_path+opt.dataName+"_G_BA"+str(epoch)+".pt"))
    # G_AB.load_state_dict(torch.load(weight_path+opt.dataName+"_G_AB"+str(epoch)+".pt"))

    # test모드로 gpu 메모리 아끼기
    G_BA.eval()
    # G_AB.eval()

    result_path='test_result_img/'+opt.dataName+'_Hybrid'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
   
   # gpu 메모리 아끼기
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            gc.collect()
            # B->A
            real_B = batch["B"].cuda().detach() # 추론과정이라 gradient 계산x
            # real_B = batch["B"]
            fake_A = G_BA(real_B)

            
            # save_image(fake_A, result_path+f"/%d.png" %(i+1), normalize=True)
            save_image(fake_A, result_path+f"/%d.%s" %(i+1, file_extension), normalize=True)

            # # B->A->B
            # real_B = batch["B"].cuda().detach() # 추론과정이라 gradient 계산x
            # # real_B = batch["B"]
            # fake_A = G_BA(real_B)
            # fake_B = G_AB(fake_A)
            
            # # save_image(fake_A, result_path+f"/%d.png" %(i+1), normalize=True)
            # save_image(fake_B, result_path+f"/%d.%s" %(i+1, file_extension), normalize=True)
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataName', type=str, default="gogh2photo_landscape", help='data name')
    parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation') #4
    opt = parser.parse_args()

    file_extension=input("저장할 그림의 확장자명을 입력 해주세요. ex) jpg png: ")

    print(opt)

    main()


