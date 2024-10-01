import argparse
import itertools
import time
import os
import gc
import torch
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


from models_LeakyReLU import GeneratorResNet
from models_LeakyReLU import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset


# from tensorboardX import SummaryWriter
# summary = SummaryWriter('loss/ink_and_wash2photo_landscape/Hybrid_256')

def Feature_Extraction(model, x):
    return model.forward(x)

def main():
    torch.cuda.empty_cache()
    gc.collect()
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches') # 256 사이즈 기준 원래 4였음 수정하기 나중에
    parser.add_argument('--dataName', type=str, default="gogh2photo_landscape", help='data name')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=64, help='size of the data crop (squared assumed)') # 64까지 일단 줄였음 느낌이 128까지는 해야할듯
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')  # 디폴트 4인데 오류 발생
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 데이터 로드
    transforms_ = transforms.Compose([
        transforms.Resize(int(opt.size * 1.12), Image.BICUBIC), # 이미지 크기를 조금 키우기
        transforms.RandomCrop((opt.size, opt.size)),
        transforms.RandomHorizontalFlip(), # 각 데이터가 단일 이미지로 존재하므로 좌우 반전 가능
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageDataset(opt.dataName, transforms_=transforms_)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    

    # 생성자(generator)와 판별자(discriminator) 초기화 block수 이미지 사이즈에 따라 256이상이면 9로 설정 128이하면 6으로 설정
    G_AB = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)
    G_BA = GeneratorResNet(input_shape=(opt.input_nc, opt.size, opt.size), num_residual_blocks=6)
    D_A = Discriminator(input_shape=(opt.input_nc, opt.size, opt.size))
    D_B = Discriminator(input_shape=(opt.input_nc, opt.size, opt.size))

    vgg_model = models.vgg16_bn(pretrained=True)

    vgg_model.cuda()
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()
    

    # 가중치(weights) 초기화
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # 손실 함수(loss function)
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_semantic = torch.nn.MSELoss()
    


    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    criterion_semantic.cuda()


    # 생성자와 판별자를 위한 최적화 함수
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A  = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # 학습률(learning rate) 업데이트 스케줄러 초기화
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.decay_epoch).step)


    lambda_cycle = 10 # Cycle 손실 가중치(weight) 파라미터
    lambda_identity = 5 # Identity 손실 가중치(weight) 파라미터
    lambda_semantic = 0.1 ############# 학습하면서 변경 필요!!!!

    # 이전에 생성된 이미지 데이터를 포함하고 있는 버퍼(buffer) 객체
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    start_time = time.time()

    # 10 에포크마다 weight 저장할 폴더 생성

    # 저장 경로 설정
    weight_path_jit='weight/jit/'+opt.dataName+'_Hybrid/'
    weight_path_torch='weight/torch/'+opt.dataName+'_Hybrid/'


    if not os.path.exists(weight_path_jit):
        os.makedirs(weight_path_jit)
    
    if not os.path.exists(weight_path_torch):
        os.makedirs(weight_path_torch)

    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(train_dataloader):
            # 모델의 입력(input) 데이터 불러오기
            real_A = batch["A"].cuda()
            real_B = batch["B"].cuda()

            ## 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성 (너비와 높이를 16씩 나눈 크기)
            # 256*256 이미지면 16으로 나눈 16 값으로, 64*64면 4로
            real = torch.cuda.FloatTensor(real_A.size(0), 1, 4, 4).fill_(1.0) # 진짜(real): 1
            fake = torch.cuda.FloatTensor(real_A.size(0), 1, 4, 4).fill_(0.0) # 가짜(fake): 0


            #생성자 학습
            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity 손실(loss) 값 계산
            loss_identity_A = criterion_identity(G_BA(real_A), real_A)
            loss_identity_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_identity_A + loss_identity_B) / 2

            # GAN 손실(loss) 값 계산
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), real)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), real)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle 손실(loss) 값 계산
            recover_A = G_BA(fake_B)
            recover_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recover_A, real_A)
            loss_cycle_B = criterion_cycle(recover_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Semantic 손실(loss) 값 계산
            # 가중치는 바꿀 필요가 있음.
            loss_semantic_A = criterion_semantic(Feature_Extraction(vgg_model, real_A), Feature_Extraction(vgg_model, recover_A))
            loss_semantic_B = criterion_semantic(Feature_Extraction(vgg_model, real_B), Feature_Extraction(vgg_model, recover_B))
            loss_semantic = (loss_semantic_A + loss_semantic_B) / 2





            # 최종적인 손실(loss)
            loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity + lambda_semantic * loss_semantic

            # 생성자(generator) 업데이트
            loss_G.backward()
            optimizer_G.step()

            # 판별자 학습
            optimizer_D_A.zero_grad()

            # Real 손실(loss): 원본 이미지를 원본으로 판별하도록
            loss_real = criterion_GAN(D_A(real_A), real)

            # Fake 손실(loss): 가짜 이미지를 가짜로 판별하도록
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)

            # 최종적인 손실(loss)
            loss_D_A = (loss_real + loss_fake) / 2

            # 판별자(discriminator) 업데이트
            loss_D_A.backward()
            optimizer_D_A.step()

            # 판별자 학습
            optimizer_D_B.zero_grad()

            # Real 손실(loss): 원본 이미지를 원본으로 판별하도록
            loss_real = criterion_GAN(D_B(real_B), real)

            # Fake 손실(loss): 가짜 이미지를 가짜로 판별하도록
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)

            # 최종적인 손실(loss)
            loss_D_B = (loss_real + loss_fake) / 2

            # 판별자(discriminator) 업데이트
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

        # 학습률(learning rate)
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 하나의 epoch이 끝날 때마다 로그(log) 출력 loss_semantic
        print(f"[Epoch {epoch}/{opt.n_epochs}] [D loss: {loss_D.item():.6f}] [G identity loss: {loss_identity.item():.6f}, G adv loss: {loss_GAN.item():.6f}, G cycle loss: {loss_cycle.item():.6f}, G semantic loss: {loss_semantic.item():.6f} final_G_loss: {loss_G.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")

    #    # 에포크마다 loss 저장
    #     summary.add_scalar('loss/loss_D', loss_D.item(), epoch)
    #     summary.add_scalar('loss/loss_identity', loss_identity.item(), epoch)
    #     summary.add_scalar('loss/loss_adv', loss_GAN.item(), epoch)
    #     summary.add_scalar('loss/loss_cycle', loss_cycle.item(), epoch)
    #     summary.add_scalar('loss/loss_semantic', loss_semantic.item(), epoch)
    #     summary.add_scalar('loss/loss_cycle', loss_cycle.item(), epoch) 
    #     summary.add_scalar('loss/loss_G', loss_G.item(), epoch) 
    #     summary.add_scalars('loss/loss_total', {"loss_G": loss_G.item(),
    #                                             "loss_D": loss_D.item()}, epoch)


        # 10에포크마다 모델 저장
        if epoch%10==0:
            torch.save(G_BA.state_dict(), weight_path_torch+opt.dataName+"_G_BA"+str(epoch)+".pt")
            torch.save(G_AB.state_dict(), weight_path_torch+opt.dataName+"_G_AB"+str(epoch)+".pt")


             # 모델구조 jlt로 전체저장
            jit_model1=torch.jit.script(G_BA)
            torch.jit.save(jit_model1,  weight_path_jit+opt.dataName+"_G_BA"+str(epoch)+".pt")

            jit_model2=torch.jit.script(G_AB)
            torch.jit.save(jit_model2,  weight_path_jit+opt.dataName+"_G_AB"+str(epoch)+".pt")
            print(str(epoch)+" epoch Model saved!")

        # 마지막 에포크 모델 저장
        if epoch==199:
            torch.save(G_BA.state_dict(), weight_path_torch+opt.dataName+"_G_BA"+str(epoch+1)+".pt")
            torch.save(G_AB.state_dict(), weight_path_torch+opt.dataName+"_G_AB"+str(epoch+1)+".pt")

            # 모델구조 jlt로 전체저장
            jit_model1=torch.jit.script(G_BA)
            torch.jit.save(jit_model1,   weight_path_jit+opt.dataName+"_G_BA"+str(epoch+1)+".pt")

            jit_model2=torch.jit.script(G_AB)
            torch.jit.save(jit_model2,  weight_path_jit+opt.dataName+"_G_AB"+str(epoch+1)+".pt")
            print(str(epoch)+" epoch Model saved!")
        
        
if __name__ == "__main__":
    main()