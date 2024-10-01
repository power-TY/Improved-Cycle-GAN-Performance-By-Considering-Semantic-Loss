import torch
import random

# 이미지 버퍼(Buffer) 클래스
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    # 새로운 이미지를 삽입하고, 이전에 삽입되었던 이미지를 반환하는 함수
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # 아직 버퍼가 가득 차지 않았다면, 현재 삽입된 데이터를 반환
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            # 버퍼가 가득 찼다면, 이전에 삽입되었던 이미지를 랜덤하게 반환
            else:
                if random.uniform(0, 1) > 0.5: # 확률은 50%
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element # 버퍼에 들어 있는 이미지 교체
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# 시간이 지남에 따라 학습률(learning rate)을 감소시키는 클래스
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs # 전체 epoch
        self.decay_start_epoch = decay_start_epoch # 학습률 감소가 시작되는 epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# 가중치 초기화를 위한 함수 정의
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)