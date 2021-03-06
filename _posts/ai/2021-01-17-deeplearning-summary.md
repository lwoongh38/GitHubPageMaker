---
layout: post
current: post
cover:  assets/built/images/SUMMARY.jpg
navigation: True
title: Deep Learning - 핵심 개념 & 용어 
date: 2021-01-17 22:00:00
tags: [ai]
class: post-template
subclass: 'post tag-ai'
author: woongE
---
#deeplearning #용어 #개념


{% include ai-table-of-contents.html %}

# Deep Learning Summary - 핵심 개념 & 용어

이제는 대세가 된 단어인 딥러닝. 대중적인 관심이 많이 늘어났지만 관련 분야를 공부를 하지 않은 일반인에겐 어려운 개념이나 용어가 많아 이해하기가 쉽지 않다. 그래서 오늘은 내가 배운 내용을 정리할 겸 일반인이 보기에도 이해하기 쉽도록 각 용어와 개념들을 설명해보려고 한다.

---
<img src="https://user-images.githubusercontent.com/70134676/104846469-0d2ee480-591e-11eb-86ee-e8bb5705f5ce.png" width="100%" height="100%">
전체적인 딥러닝 개요도

## 1. 신경망 기초
<img src="https://user-images.githubusercontent.com/70134676/104845364-60059d80-5918-11eb-8e0f-d48adb341dd8.png" width="80%" height="80%">


(1) 뉴런  
인공신경망에서 세포의 역할을 하고 신경망 구조도에서는 노드로 표현된다. 
뉴런마다 각각 입력을 받게되고 활성함수를 거쳐 다음 뉴런으로 출력한다. 

(2) weight(가중치)  
인공신경망에서 뉴런을 연결하는 선으로 각 가중치들은 입력받은 신호를 필터링하는 역할을 한다.
입력받은 값에 어느정도의 가중을 주어 다음 뉴런에 전달할 것인지 해당 데이터의 경중을 결정하는 요소이다.

(3) bias(편향)  
뉴런들마다 각각 입력된 값과 가중치를 곱한 다음 더해주는 상수이다. 각 뉴런이 가지고있는 고유값이다.

(4) 활성화함수  
하나의 뉴런에서 다음 뉴런으로 가중합을 전달할 때, 값을 전달할지 말지를 결정하는 기준이 되는 함수이다. 임계점을 기준으로 출력값에 변화를 주게된다.

<img src="https://t1.daumcdn.net/cfile/tistory/994C83365D4FD0D125" width="80%" height="80%">

- sigmoid  
이진분류를 할 때 사용되며 기본적으로 임계값은 0.5로 이 값을 기준으로 작으면 0으로, 크면 1로 분류한다.
- ReLu  
0을 기준으로 0보다 작을 때는 0을, 클 때는 입력값 자체를 출력하는 기울기가 1인 함수
- softmax  
다중분류문제에 사용되며 0~1 사이의 값을 출력하며 확률값이기 때문에 전부 더하면 1이 된다.
- Tanh  
시그모이드의 변형버전으로 임계값을 기준으로 -1, 1로 분류한다.

## 2. 신경망 학습

(1) 역전파 & 경사하강법  
실제 관측값과 예측값 사이의 오차가 최소가 되도록 역전파 알고리즘은 순전파와 반대로 거슬러올라가며 경사하강법을 사용하여 가중치를 업데이트한다.

<img src="https://user-images.githubusercontent.com/70134676/104846115-4f572680-591c-11eb-85ce-a559d3f55d7f.png" width="80%" height="80%">


<img src="https://user-images.githubusercontent.com/70134676/104846131-6138c980-591c-11eb-8d1d-183b1c274987.png" width="80%" height="80%">


<img src="https://user-images.githubusercontent.com/70134676/104846360-6c402980-591d-11eb-8643-190f357bdaab.png" width="80%" height="80%">

(2) 손실함수
- 분류  
  - binary cross-entropy  
    이진분류할 때 사용하는 함수로 0에 가까울수록 좋다.  
  - categorical cross-entropy  
    다중분류에 쓰이는 함수로 0에 가까울 수록 좋다.  
  - sparse cross-entropy  
    categoricalCrossentropy 와 비슷하지만, 레이블이 int 형이라는 점에서 다르다.
 
- 회귀
  - MAE  
    관측값과 예측값의 차이에 절대값을 취하여 모두 합한 후 평균을 계산한 값이다. 실제값과 단위가 똑같아 직관적인 비교가 가능하다.  
  - MSE  
    관측값과 예측값의 차이를 제곱하여 평균을 계산한 값이다. MAE보다 아웃라이어에 민감하다.

(3) optimizer  
- SGD  
  경사하강법은 lost function를 미분을 통하여 최소값을 찾아가는 과정이다. 경사하강법 에서는 학습률를 중요하게 생각해야하는데, 이것이 너무 작을땐 탐색시간이 한참걸릴수도 있고, 너무 클땐 최소 손실값을 잘못 찾을수도 있기에 (전역 최솟값을 건너뛰고 지역 최솟값을 찾는 문제) 비교적 비효율적이다. 그래서 SGD는 GD 와 다르게, 한번 학습할 때 모든 데이터에 대해 가중치를 조절하는 것이 아니라, 랜덤하게 추출한 일부 데이터에 대해 가중치를 조절한다.
- RMSprop  
  AdaGrad 은 학습률을 변수들에 다르게 적용하는 방법이다. 변화가 많았던 변수들엔 학습률을 작게하고, 그렇지 않은 변수들엔 크게 하는것이다. 많은 변화가 있었다는 것은 최적값에 가깝다는 뜻이니 학습률을 작게해서 세밀하게 조정하기 위함이다. 하지만, AdaGrad 로인해 너무 극단적으로 학습률이 작아질때 생기는 학습이 멈추는 문제가 있었는데 이것을 보완한 것이 RMSprop 이다.
- Adam  
  Momentum 은 GD 의 학습률로 인한 문제를 기울기의 정도에따라 관성을 적용함으로써 완화한 것이다. Adam 은 Momentum 과 RMSprop 이 가지는 장점을 동시에 가지는 컨셉으로 생겨났다.
  
(4) 학습 규제
- dropout  
  노드를 랜덤하게 일시적으로 끊어낸다. 해당 뉴런을 물리적으로 차단하여 학습을 실시, 과적합을 방지한다.
- early stopping  
  학습을 진행하면서 사용자가 정한 임계값 이상의 성능 개선이 없으면 학습을 조기종료하여 과적합을 방지한다.
- weight decay  
  가중치가 커지는 것을 방지하도록 가중치를 감소시키는 방법으로 과적합을 방지한다.
- learning rate decay  
  학습할수록 학습률을 감소시켜 과적합을 방지한다.

(5) 메모리 사용 전략  
- batch  
  모델의 가중치를 업데이트 시킬 때 사용되는 데이터의 단위를 말한다. 1000개로 구성된 데이터가 있을 때, 배치 사이즈를 100으로 설정하면 100개 단위로 가중치를 업데이트 한다. 전체 데이터에 대하여 총 10번의 가중치가 업데이트된다.
- epoch & iteration
  학습 횟수를 의미하며 1000개로 구성된 데이터에서 batch size가 10이고 epoch이 10이면 100번의 가중치 업데이트를 한 사이클로 하여 총 10 사이클을 반복하여 실시한다. 이때 100번의 가중치 없데이트 횟수가 iteration이 된다. 1epoch은 데이터 전체를 한번 사용하는 것을 기준으로 한다.

## 3. CNN

(1) convolution  
합성곱을 통하여 fully connected 신경망보다 가중치의 갯수가 줄어들어 결과적으로 학습시간이 훨씬 빠르고 컴퓨팅 파워를 절약시켜준다.  
- stride  
  filter를 움직이는 보폭을 의미한다. stride가 1이면 filter를 한칸씩 움직이며 특징을 추출한다. stride가 작을수록 세세하게 필터링하게된다.
- padding  
  채널은 필연적으로 입력데이터보다 작아지게 되는데 특성추출이 반복되어 차원이 0이 되는 것을 방지하는 방법이다. 데이터 가장자리에 0, 평균값 등 다양한 특정값을 채워넣어 차원축소를 방지한다. 패딩을 해주면 데이터를 골고루 사용하게 되고 합성곱 연산 이후에도 차원이 유지된다.
- filter  
  weight들의 집합으로 데이터의 특징을 추출하는 window 역할을 한다.

(2) pooling  
채널의 피처맵 차원을 축소하는 방법으로 합성곱 연산 이후 레이어를 풀링하여 더 작은 피처맵을 추출한다.  

## 4. RNN

- RNN  
  은닉층의 출력데이터가 다시 자기자신 노드로 순환되어 입력값으로 사용된다. 이런 특성 때문에 시계열 데이터를 다루는 데에 좋은 성능을 낸다. 예를 들어, 주가데이터, 음성데이터, 텍스트데이터 같이 앞 뒤 데이터간의 연관성이 있는 데이터셋에 사용될 수 있다.

- LSTM  
  RNN의 문제 중 하나는 시계열 데이터를 다룰 때, 정보의 위치가 멀수록 역전파를 할 때 경사가 0에 수렴하기 때문에 모델 성능에 악영향을 미친다는 것이다. 
  "나는 한국에서 한국어를 배운다"라는 문장에서 "한국어"라는 단어를 예측한다고 할때 문장에서 핵심적인 정보를 제공하는 단어는 "한국"이다. 이처럼 "한국"과 "한국어"의 위치가 가까이에 있다면 문맥을 연결하기가 쉽지만 멀리 위치해있다면 문맥을 연결하기 힘들어 성능이 저하된다. 역전파 과정에서 과거와 현재의 거리가 멀어질수록 gradient 값이 소실되는 gradient vanishing 문제가 대두되어 LSTM의 개념이 고안되었다.  
  이전 정보를 얼마나 잊을지, 현재 정보를 어느정도로 반영할지, 정보를 밖으로 얼마나 출력할지를 결정하는 Gate를 추가하여 RNN을 보완한다.

- GRU  
  LSTM의 게이트 숫자를 줄여 모델을 구조적으로 단순화했다는 점이 다르다. forget과 input gate를 합치고 output gate를 생략하였다.

## 5. GAN

- DCGAN  
  생성모델의 일종으로 실제로 값어치가 있는 그럴싸한 데이터를 생성해내는데 그 존재 이유가 있다. 생성자 모델과 감별자 모델 두가지가 서로 대립하여 서로 경쟁적으로 학습하여 모델의 성능을 향상시키는 컨셉이다.

- CycleGAN  
  데이터의 원래 형태는 유지하면서 세부 특성만 교체할 수 있게 하는 모델이다. 각각 두 개의 생성자와 감별자를 두어 특성을 추출한다. (말, 얼룩말), (오렌지, 사과) 이미지 쌍을 입력하여 학습을 할 때는 말을 얼룩말으로 바꾼 이미지는 얼룩말인지 아닌지 판단하여 얼룩말이라고 판단되도록, 또 변경된 이미지를 다시 말로 바꿔서 원본과 비교함으로써 다른 속성들은 그대로 유지되었는지 판단하며 그 차이를 줄이는 방향으로 학습한다.

## 6. Autoencoder

오토인코더는 인코더와 디코더로 구성되어있다. 인코더는 입력 데이터를 압축하여 벡터화 하게되고 디코더는 압축된 벡터를 다시 원본데이터와 유사하게 복원하는 역할을 한다.  
원래의 입력데이터 x와 압축, 복원을 거친 x'의 차이를 손실로 정의하고 손실이 줄어들 수 있도록 역전파를 통해 학습이 이루어진다. 입력데이터를 벡터화하는 과정에서 입력 데이터를 정의할 수 있는 잠재변수가 만들어지는데 이 잠재변수의 크기가 커지면 데이터를 구분할 수 있는 더 다양한 특징들이 추출된다. 반대로 잠재변수의 크기를 줄이면 특징의 수는 줄어들지만 더 핵심적으로 데이터를 구분할 수 있는 특징들이 추출된다.

- Encoder : 입력 데이터의 특징을 담고있는 Latent 로 데이터를 압축시키는 단계.

- Decoder : 압축된 Latent 를 입력데이터와 유사하게 출력하기위해 확장하는 단계.

- 오토인코더의 용도
  - 노이즈 제거
  - 특성 추출

## 7. Attention

- Transformer  
자연어처리 분야에서 사용되는 기술로 문장 데이터를 한번에 입력하고 순서정보를 입력해 번역하는 모델. 크게 음성을 텍스트로 변환하거나 텍스트를 음성으로, 또는 텍스트를 다른 언어로 번역하는 모델 등이 있다.  

- BERT  
![image](https://user-images.githubusercontent.com/70134676/104848544-cbf00200-5928-11eb-8dc2-faf07c786b37.png)

구글에서 개발하고 사전훈련시킨 자연어처리에 사용되는 범용 언어모델이다. 총 33억 개의 코퍼스(bookcorpus : 8억, wikipedia : 25억)를 사용하여 개인이 구축하기 힘든정도의 방대한 양의 코퍼스(의미뭉치)로 상당한 시간 사전학습을 마쳐놓아 상당한 성능의 모델을 바로 가져다 쓸 수 있게 구현 해놓다.

자연어를 처리할 때, 데이터가 충분하다면 Embedding 과정이 성능에 큰 영향을 미치는데 이 말은 단어의 의미를 잘 간직할 수 있도록 벡터로 표현하는 것이 중요하다는 의미이다. 이 Embedding 과정에 구글의 강력한 BERT를 이용하는 것이고 사용하고자 하는 문제에 맞도록 파인튜닝을 진행한 후에 분류에 적용하게 된다. 
