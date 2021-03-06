---
layout: post
current: post
cover:  assets/built/images/linearregression.png
navigation: True
title: 연어의 회귀본능이 아닌 선형회귀에 대해 알아보자
date: 2020-10-16 19:00:00
tags: [ai]
class: post-template
subclass: 'post tag-ai'
author: woongE
---
#선형회귀 #linearregression


{% include ai-table-of-contents.html %}

# 연어의 회귀본능이 아닌 선형회귀에 대해 알아보자

선형회귀? 그게 뭐야? 회귀라는 말을 연어를 통해서만 들어본 그대.
말이 어렵지 사실 그 개념 자체는 이미 우리가 알고있는 것이나 마찬가지다. 최대한 쉽게쉽게 풀어서 설명해볼 터이니
오늘도 시작해보자.

선형회귀라고 대놓고 주제를 던졌으니 의미를 먼저 알아야겠지?
y = ax+b 라는 식이 있다면 입력항인 x와 출력항인 y의 선형 상관관계를 모델링하는 기법이다. 
아직도 모호한가? 그러면 이 말을우리 생활 속의 예제로 녹여보자.

![image](https://user-images.githubusercontent.com/70134676/96222124-a1f41000-0fc6-11eb-8d00-94644528b877.png)


저 위의 `y = ax+b`를 시험점수에 대한 식이라고 예를 들면,
- y = 물가
- x = 휴지가격
- a = 가중치(운송비용 등)
- b = 기본물가

이렇게 정리할 수 있다.
위의 그림대로라면 휴지가격을 알 때 물가가 어떠한지를 얻을 수 있다.
이것도 간단한 회귀식이다. 이처럼 수식화하지 않았을 뿐 우리 생활에는 다양한 회귀식들이 숨어있다.
그럼 여태까지 회귀식을 모르고도 사는데 전혀 문제가 없었는데 갑자기 왜 들고나와서 머리아프게 하는지 궁금해 하는 사람이 있을 것이다.
문제는 이처럼 간단한 회귀식이라면 상관이 없겠지만 데이터의 수가 엄청나게 많아지고 특징들이 많아진다면 사람의 머리로는 저런 생각을 하는 것이 불가능 할 것이다. 이를 위해 우리는 계산기(컴퓨터)의 힘을 빌리게 되고 컴퓨터의 좋은 성능을 이용하면 알고있었던 데이터(빨간 점)에서 답을 찾는 것 뿐만이 아닌, 새로운 데이터(x축, 휴지가격)을 바탕으로 정답(y축, 물가)를 예측해볼 수도 있다.
이것이 요즘 흔히 하는 말로 **머신러닝**이라고 하는 개념이다.
물론 머신러닝도 지도, 비지도 등 여러가지 갈래로 나뉘어지지만 위에서 설명한 머신러닝은 선형적인 관계를 가진 데이터를 위한 선형회귀 예측모델이 되는 것이다. 빨간점을 활용하여 학습시키면 다른 데이터(이지만 비슷한 선형관계를 가진) 파란점의 휴지가격을 알 때, 물가를 예측해볼 수 있는 것이다. 
한마디로 **선형회귀는 선형관계를 가진 데이터를 모델링하여 예측하기 위한 머신러닝의 일종**이다.

선형회귀 기법을 사용하여 데이터에 맞는 예측모델을 만들기 위해서는 회귀직선 필요하다.
![image](https://user-images.githubusercontent.com/70134676/96222090-97397b00-0fc6-11eb-8bdf-47ef8c0e2a32.png)

회귀직선이 말하고자 하는 바는 예측을 할 때 이러한 느낌으로 예측을하면 맞출 수 있을것이라는 일종의 경향성을 의미하고 위의 그림에서는 녹색선이 그 경향을 의미한다.  우리는 데이터로 빨간 점을 가지고 있고 이 데이터가 어떤 경향을 띄는지를 가장 잘 설명해주는 선이 바로 녹색 선이다. 

![image](https://user-images.githubusercontent.com/70134676/96223864-9524eb80-0fc9-11eb-8a0c-ffbd1840ad96.png)
이 선을 설명하기 위해서는 예측값과 잔차라는 개념을 알아야 하는데 예측값은 저 녹색선이 추정하는 값이고, 잔차는 관측값(빨간점의 물가)와 예측값(파란점의 물가)의 차이를 말한다.
녹색선은 그림의 검은점선으로 표시된 저 관측값과 파란점선의 예측값의 제곱의 합이 최소가 되어야 가장 경향을 잘 설명할 수 있다.
여기서 파란점선의 크기와 검은 점선 크기의 제곱의 합을 잔차제곱합이라고 하는데 이 잔차제곱합을 비용함수(cost function)라고 하는데 머신러닝 모델을 만드는 과정에서 비용함수를 최소화 하는 모델(선)을 찾는 과정을 학습이라고 한다.

저 녹색 선을 그려서 저 녹색선을 바탕으로 값을 예측하게 하는 과정 전반이 머신러닝 선형회귀 모델링이 되는 것이다.

선형회귀 모델은 주어져있지 않은 함수값을 보간하여 예측하는데 유용한데 예를들어 빨간점이 아닌, 없는 데이터 즉 파란점에 해당하는 휴지가격을 알고 있을 때 원래 데이터(빨간점)에는 없지만 녹색선을 이용하여 파란점의 휴지가격을 알 때  물가를 어림잡아 예측해볼 수 있는 것이다.

오늘은 머신러닝의 많은 모델 중 하나인 선형회귀에 대해서 알아보았다. 이제는 선형회귀에 대한 말이 나오면 자신있게 대화에 끼어들어보자.
앞으로도 다양한 머신러닝의 모델들을 알기쉽게 소개해보리라고 다짐하며 글을 마친다.



