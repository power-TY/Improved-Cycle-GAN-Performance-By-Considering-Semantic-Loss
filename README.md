# Improved-Cycle-GAN-Performance-By-Considering-Semantic-Loss
Improved Cycle GAN Performance By Considering Semantic Loss

# Abstract
Recently, several generative models have emerged and are being used in various industries. Among them, Cycle GAN is still used in various fields such as style transfer, medical care and autonomous driving. In this paper, we propose two methods to improve the performance of these Cycle GAN model. The ReLU activation function previously used in the generator was changed to Leaky ReLU. And a new loss function is proposed that considers the semantic level rather than focusing only on the pixel level through the VGG feature extractor. The proposed model showed quality improvement on the test set in the art domain, and it can be expected to be applied to other domains in the future to improve performance.

# Model Architecture
![image](https://github.com/user-attachments/assets/7c176773-6a7d-4b89-98d3-f465fbc8543c)

# Loss Term
![image](https://github.com/user-attachments/assets/1ea8b027-56dc-4894-8554-216905cc5b9e)

# Experiments

## Fid Score
![image](https://github.com/user-attachments/assets/2d15b310-60ee-487e-9fa2-c789f562a513)

## Generated Image
![image](https://github.com/user-attachments/assets/a42cf2ab-20e9-4dc8-b195-e3ecfeb95ab9)
![image](https://github.com/user-attachments/assets/38ef37d8-0198-495e-932f-aa033aa0a3e8)
![image](https://github.com/user-attachments/assets/57c8a2a3-4fe2-47fb-a6d3-fde04e5b5b71)
![image](https://github.com/user-attachments/assets/e1a06df3-03fa-4f81-8c4e-982bfbeb0e93)

# Paper
정태영, et al. "의미적 손실 함수를 통한 Cycle GAN 성능 개선." 한국정보처리학회 학술대회논문집 30.2 (2023): 908-909.



