---
layout: post
current: post
cover:  assets/built/images/tadgan.png
navigation: True
title: Time seriesAnomaly Detection Generative Adversarial Networks-Model Structure
date: 2021-11-27 22:00:00
tags: [ai]
class: post-template
subclass: 'post tag-ai'
author: woongE
---
#deeplearning #TadGAN #modelstructure


{% include ai-table-of-contents.html %}

# Time series Anomaly Detection Generative Adversarial Networks - Model Structure


## 1. TadGAN의 구조

지금까지 우리는 TadGAN이 어떤 모델이고, 어떤 loss function을 쓰는 지 알아봤습니다. 하지만 우리는 가장 중요한 걸 모르고 있습니다. 바로 모델 그 자체가 어떻게 작동하는지 입니다. 논문에는 tadgan모델 구조를 다음과 같이 나타냈습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630537-6932bae6-fe81-4b10-82e3-a65c1e2cc886.png" width="60%" height="80%"></p>

<br/>

모델은 기본적으로 AutoEncoder의 구조와 유사한 구조를 가지고 있습니다. 두 개의 Generator, 두개의 Critic 함수로 이루어졌다는 것을 확인할 수 있습니다. Generator E는 time series sequence를 latent space로 바꿔주는 역할을 하고, Generator G는 latent space를 다시 time series sequence로 변환해주는 역할을 합니다. 여기까지는 기존의 AE와 별반 다를 것이 없어 보입니다. 하지만 Critic X함수는 원본 데이터와 생성된 데이터 중 어떤 데이터가 원본인지를 가려내고, Critic Z 함수는 Generator E가 time series squence를 latent space로 얼마나 잘 맵핑 했는지를 판별 합니다. 더 자세하게 모델이 어떻게 작동하는지를 알기 위해서 TadGAN 공식 github의 이미지를 가지고 설명을 할까 합니다. 모델 이해는 pytorch에 능숙하다는 가정하에 (https://github.com/arunppsg/TadGAN)를 보시는 것을 추천드립니다.

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630614-58377363-5d81-41bb-8306-e12e7e496b62.png" width="60%" height="80%"></p>

<br/>

전체 모델 구조입니다. E,G는 Encoder과 Decoder를 나타내고, Cx를 통해서 실제 데이터와 Reconstruction 데이터를 판별합니다. 그리고 Cz는 E를 통해 생성된 Latent space와 렌덤으로 생성된 데이터를 가지고 얼마나 잘 맵핑되어 있는 지를 판별합니다. 다음 부터는 각 Phase 별로 어떤 학습이 이루어지는지 볼텐데, 우리가 주목할 것은 학습의 주체가 무엇인지 입니다.

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630638-afe2af36-81d0-4b87-b455-1c5e680989e3.png" width="60%" height="80%"></p>

<br/>

Phase 1 입니다. 실제 데이터와 random latent space가 Generator G를 통과한 재생성 데이터를 판별하는 Cx를 훈련 시키게 됩니다.

Cx Loss function = critic_score_fake_x - critic_score_valid_x + sqrt[sum(Cx(alpha * x + (1 - alpha) * x_)^2)]

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630664-00de5c7f-d4a4-4f24-897a-707be22821a3.png" width="60%" height="80%"></p>

Phase 2 입니다. 여기서는 실제 데이터를 맵핑 시킨 latent space와 random latent space를 비교해 Cz가 어떤 데이터가 원본 데이터의 latent space인지 판별하도록 학습합니다.

Cz Loss function =critic_score_fake_z - critic_score_valid_z + sqrt[sum(Cx(alpha * z)+ (1 - alpha) * z_)^2)]

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630687-52c09ef8-2511-4a22-815c-8c108a351bfd.png" width="60%" height="80%"></p>

Phase 3 입니다. 실제 데이터에서 나온 latent space를 가지고 Generator G로 재생성한 time series squence(gen_x)와, random latent space를 가지고 Genrator G로 생성한 time series squence (fake_x)를 Cx로 판별해 실제 데이터를 구별합니다. 이 때 학습 주체는 Generator E 이고, Cx를 최대한 속이는 쪽으로 학습을 진행합니다. 그리고 원본데이터와 random latent space로 생성한 fake_x 데이터의 Cx Score들을 loss function에 추가했는데, Decoder의 학습 정도에 따라 학습 속도를 변화시킨 것을 볼 수 있습니다.

Generator E(Encoder) Loss function =mse_loss(x,gen_x)+critic_score_valid_z - critic_score_fake_z

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/152630709-4aa325f3-01d8-4ba1-bdf0-093727a1b6ff.png" width="60%" height="80%"></p>

<br/>

Phase 4 입니다. 실제 데이터를 Generator E에 통과 시킨 latent space를 Generator G에 통과시킨 gen_x와 실제 데이터 x를 비교합니다. 이 때 원본 데이터의 latent space와 random latent space를 통과시킨 Cz의 score들을 loss function에 추가했는데, Phase3 처럼 Encoder의 학습 정도에 따라 학습 속도를 변화시킨 것을 볼 수 있습니다.

Generator G(Decoder) Loss function =mse_loss(x,gen_x)+critic_score_valid_x - critic_score_fake_x

위의 과정은 차례대로 encoder_iteration,decoder_iteration critic_x_iteration, 그리고 critic_z_iteration로 소스코드에 포함되어 있습니다. 코드를 보고 천천히 따라가시면 어떻게 모델이 작동하는 건지 알 수 있습니다.

