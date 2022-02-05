---
layout: post
current: post
cover:  assets/built/images/tadgan.png
navigation: True
title: Intro to the Time Series Anomaly Detection
date: 2021-11-19 22:00:00
tags: [ai]
class: post-template
subclass: 'post tag-ai'
author: woongE
---
#TadGAN #Timeseries #AnomalyDetection 


{% include ai-table-of-contents.html %}

# Time series Anomaly Detection Generative Adversarial Networks


## 1. Intro  
<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150893257-5d606456-8c67-4801-84e4-7a82dd906dd9.png" width="80%" height="100%"></p>

<br/>

이번에 TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks
논문을 읽고 공부하면서 알게 된 내용을 정리하는 포스트입니다. 한 포스트에 담기에는 양이 너무 많을 것 같아 2~3개 정도의 시리즈로 나누어 정리해볼 생각입니다.
TadGAN은 machine learning 중 비지도 학습 방식입니다. 입력되는 시계열 데이터와 최대한 유사한 데이터를 모방하도록 학습하고, Threshold에 따라 이상치를 판단하는 이상 탐지 기법입니다. 비지도 학습 이상 탐지 분야는 다양한 목적을 위해 활용되고 있는데 다음과 같이 열거할 수 있습니다.  

<br/>

### 데이터 품질 향상
스탠포드 대학의 Andrew ng 교수는 모델을 튜닝하는 것 못지않게 모델의 학습에 이용되는 데이터의 품질을 높이는 것이 모델 성능 향상에 주요한 영향을 미친다고 말했습니다. 따라서 신뢰할 수 있는 데이터를 수집해야 하는데 이 과정에서 데이터 품질을 높이기 위해 필터링하는 작업에 이상치 탐지 모델을 적용하려는 노력이 이어지고 있습니다.  

<br/>

### 비지도 학습방법의 필요성(Label의 부재)
현실에 존재하는 데이터의 대부분은 독립변수에 따른 종속변수인 Y Label을 알 수 없는 경우가 많습니다. 이러한 데이터에 Labeling 작업을 통해 지도학습방식으로 모델을 학습시킬 수도 있지만 굉장히 많은 비용(시간, 돈)이 소모될 뿐만 아니라 Labling을 하는데 필요한 기준을 세우는 것도 어렵습니다. 그래서 결국에는 비지도학습에 의한 이상치 탐지방법이 필요합니다.

이상치 탐지를 하는 방법은 여러가지가 존재하며 저는 그 중에서도 논문에서 소개한 세가지 방법론을 본격적으로 TadGAN을 얘기하기 전에 소개 하려고 합니다.    
<br/>

### 1. Proximity-based methods  

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150920567-fb0236b3-1a25-41b5-9521-3c1d2273f29f.png" width="80%" height="100%"></p>

<br/>

첫번째는 근접도 기반 방법입니다. 대표적으로 clutering 기법이 존재합니다. 각 데이터 요소들간의 거리를 계산하고 이를 바탕으로 근접도를 판단을 합니다. 근접도가 커지면 커질수록 멀리 있다는 뜻이기 때문에 여러가지 제약조건을 통해서 outlier을 결정할 수 있습니다.  
<br/>

### 2. Predict-based methods  

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150920713-9916bf4f-c3f3-4cef-b76f-c06c1a9a21ca.png" width="80%" height="100%"></p>

두번째는 예측 기반 방법입니다. 전통적인 통계방법인 ARIMA가 가장 대표적입니다. 시점 t-1까지의 데이터를 활용해서 시점 t를 예측합니다. 그리고 실제 값이 예측값과 얼마나 다른지를 보고 이상치인지 아닌지를 판별합니다. 세 가지 방법중에서는 난이도가 가장 높은 편이라고 할 수 있습니다.  
<br/>

### 3. Reconstruction-based methods

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150920953-f5d3a489-6ac8-4a36-a207-c2201c291ff1.png" width="80%" height="100%"></p>

<br/>

재생성 기반 방법입니다. Auto Encoder가 가장 대표적입니다. 여러개의 특징(다변량 혹은 시계열데이터)을 Encoder을 통해 latent space로 압축시키고, 이후 Decoder을 통과해 원래의 데이터로 복원하게 되는데 이때 Auto Encoder는 그 데이터의 특징을 학습한 상태이므로 이상치가 포함된 데이터라면 복원이 잘 되어있지 않을 것입니다. 이러한 원리로 복원된 데이터와 원본 데이터의 차이를 비교해서 이상치를 추정하게 됩니다.  
<br/>

## 그래서 TadGAN은...

<br/>

위의 방법중에서 TadGAN은 마지막 방법인 Reconsturction-based methods를 통해 이상치를 판별합니다. 조금 더 나아가서 TadGAN은 이런 reconsturction method를 CycleGAN에 사용된 기법을 이용하는 모델입니다. 먼저 CycleGAN에 대해서 이야기해 보자면 CycleGAN은 최초로 등장했을 때 이미지에 대한 변환을 주로 담당했는데, 원본 데이터의 형태를 유지하면서 이미지의 질감이나 스타일을 잘 바꿔줍니다. 예를들면, 초원에서 뛰어다니고 있는 말 사진에서 말을 얼룩말로 바꿔줄 수 있습니다. 중요한 것은 CycleGAN은 이미지의 형태가 완전히 다르면 제대로 기능을 하지 못한다는 것입니다.  
<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150922003-011f01b4-d479-4909-a577-382459d39f9b.png" width="80%" height="100%"></p>
  
<br/>

예를 들면 말에 사람이 타고 있다면 모델은 사람도 말로 인식해서 얼룩말 무늬를 씌워버립니다. 이 것은 이미지를 학습하는 입장에서는 별로 달갑지 않겠지만 시계열 데이터가 속해 있는지 아닌지를 판별할 때는 굉장히 유용합니다. CycleGAN은 AutoEncoder와 굉장히 유사하지만 GAN으로 학습하기 때문에 조금더 폭넓은 데이터들, 그러니까 초원과 말(들)이 존재하는 모든 사진을 하나의 필드로 인식합니다. 이것은 저희가 TadGAN에서 하려는 것과 굉장히 유사합니다. 다양하지만 하나의 도메인에서 비롯된 시계열 데이터를 학습시키면 그 모델은 특정 도메인에 대해서만 좋은 복원률을 보이고, 아니라면 완전히 이상한(ex. 얼룩무늬로 이루어진 사람)복원을 하게 될 것입니다.  

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150922354-27418052-342f-41b8-8863-bc29a24c2134.png" width="80%" height="100%"></p>

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150922442-52fba2ab-6640-468c-9dd5-c8553dcdebec.png" width="80%" height="100%"></p>

<br/>

그래프를 보면 조금 더 쉽게 비교가 가능합니다. 처음 볼 수 있는 그림은 cyclegan의 원본을 시계열 데이터에 맞춰서 변경한 것입니다. 원본 시계열과 재생성된 시계열을 검사하는 Cx가 추가된 것 말고는 굉장히 흡사하게 보입니다. 우리가 여기서 하고 싶은 것은 명확합니다. 앞서 설명드렸던 cycleGAN의 아이디어를 TadGAN으로 가져와서 생각해보면 원본 이미지는 우리가 가지고 있는 시계열 데이터를 의미합니다. 우리가 가장 중요하게 생각하는 것은 시계열 데이터의 다양한 형태를 모델이 익히는 것인데 이미지로 치환해서 생각해보자면 우리가 관심 있는 것은 이미지의 형태이지 질감이나 스타일이 아닙니다. 질감이나 스타일은 여러가지 조건에 따라 달라질 수 있지만, 형태가 달라져버리면(말과 초원만 있는 사진을 학습한 모델에 말을 탄 사람이 초원에 있는 사진을 물어본 경우) 굉장히 다른 형태(얼룩무늬 사람)로 복원을 하기 때문입니다. TadGAN은 바로 이런 점에 착안을 해서 고안된 모델입니다.

이번 포스트에서는 이정도로 하고 다음 포스트부터 TadGAN의 구조나 원리 등에 대해 제가 이해한바를 적어볼 예정입니다. 제가 이해한 바가 틀릴 수도 있으니 잘못알고 있는 것이 있다면 지적해주시면 감사하겠습니다.

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/70134676/150922354-27418052-342f-41b8-8863-bc29a24c2134.png" width="80%" height="100%"></p>

<br/>

