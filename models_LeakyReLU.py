
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            # nn.LeakyReLU(inplace=True), # 새롭게 변경
            nn.ReLU(inplace=True), # 논문 원본
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)



class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0] # 입력 이미지의 채널 수: 3

        # 초기 콘볼루션 블록(Convolution Block) 레이어 c7s1-64
        out_channels = 64
        model = [nn.ReflectionPad2d(channels)]
        model.append(nn.Conv2d(channels, out_channels, kernel_size=7))
        model.append(nn.InstanceNorm2d(out_channels))
        model.append(nn.LeakyReLU(inplace=True)) # 새롭게 변경
        # model.append(nn.ReLU(inplace=True)) # 논문 원본
        in_channels = out_channels

        # 다운샘플링(Downsampling) d128, d256
        for _ in range(2):
            out_channels *= 2
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)) # 너비와 높이가 2배씩 감소
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.LeakyReLU(inplace=True)) # 새롭게 변경
            # model.append(nn.ReLU(inplace=True)) # 논문 원본
            in_channels = out_channels
        # 출력: [256 X (4배 감소한 높이) X (4배 감소한 너비)]

        # 인코더와 디코더의 중간에서 Residual Blocks 사용 (차원은 유지)
        # 이미지 크기 256이상이면 블록 수 9개, 이하면 6개
        for _ in range(num_residual_blocks):
            model.append(ResidualBlock(out_channels))

        # 업샘플링(Upsampling) u128, u64
        for _ in range(2):
            out_channels //= 2
            model.append(nn.Upsample(scale_factor=2)) # 너비와 높이가 2배씩 증가
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)) # 너비와 높이는 그대로
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.LeakyReLU(inplace=True)) # 새롭게 변경
            # model.append(nn.ReLU(inplace=True)) # 논문 원본
            in_channels = out_channels
        # 출력: [256 X (4배 증가한 높이) X (4배 증가한 너비)]

        # 출력 콘볼루션 블록(Convolution Block) 레이어 c7s1-3
        model.append(nn.ReflectionPad2d(channels))
        model.append(nn.Conv2d(out_channels, channels, kernel_size=7))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # 콘볼루션 블록(Convolution Block) 모듈 정의
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)] # 너비와 높이가 2배씩 감소
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False), # 출력: [64 X 128 X 128]
            *discriminator_block(64, 128), # 출력: [128 X 64 X 64]
            *discriminator_block(128, 256), # 출력: [256 X 32 X 32]
            *discriminator_block(256, 512), # 출력: [512 X 16 X 16]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1) # 출력: [1 X 16 X 16]
        )
        # 최종 출력: [1 X (16배 감소한 높이) X (16배 감소한 너비)]

    def forward(self, img):
        return self.model(img)