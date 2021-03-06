---
layout: post
current: post
cover:  assets/built/images/twitterlogo.jpg
navigation: True
title: Twitter Web Application
date: 2020-12-13 16:40:00
tags: [softwareengineering]
class: post-template
subclass: 'post tag-softwareengineering'
author: woongE
---
#Twitter #application #softwareengineering


{% include softwareengineering-table-of-contents.html %}

# Section 3 Project - Twitter Application

Software Engineering 섹션의 마무리 프로젝트 주제는 Flask를 이용하여 웹 어플리케이션 만들기. 트위터 API를 활용하여 데이터를 받아오고 DB를 구축하여 저장, 다시 저장된 데이터를 불러와 각종 기능들을 구현하는 것이 최종 목표였다.
섹션 내내 나를 혼돈속으로 밀어넣었던 각종 개발 툴들이 어떻게 활용되고 그것들의 퍼즐을 맞춰 최종적으로 어떤 어플리케이션이 탄생했는지 소개하려고 한다.

[Github repo](https://github.com/lwoongh38/portfolio "프로젝트링크") 


## 무슨 기능을 가진 어플리케이션인가??
---

트위터 어플리케이션은 메인페이지와 다섯가지의 기능을 가진 각 페이지를 합쳐 총 여섯개의 페이지로 구성되어 있다.
각각의 기능을 살펴보면,

- Home : 가장 처음 접속할 때 보이는 대문과 같은 페이지로 아래에 기술될 기능들을 사용하며 데이터베이스가 구축되면 user 테이블 전체를 쿼리해서 확인할 수 있도록 구성되어 있다.
  
- Add : 어플리케이션에서 정보를 조회하고싶은 트위터 유저의 screen name을 입력하면 해당 유저의 ID, Username, Full Name, Location을 API를 통해 받아와서 데이터베이스의 User 테이블에 저장한다. 또한 해당 유저의 트윗 기록(ID, Text, User ID)도 Tweet 테이블에 저장하며 User 테이블에 저장된 내용을 페이지 하단부에 쿼리하여 출력한다.

- Get : Add페이지에서 조회하여 저장된 유저의 Tweet 테이블을 쿼리하여 하단부에 출력한다.

- Delete : 데이터베이스에 저장된 유저의 User, Tweet 테이블의 데이터를 삭제할 수 있다.

- Update : 저장된 데이터베이스에서 유저의 Full Name을 변경할 수 있다.

- Predict : 트윗 내용을 제시했을 때, 두 명의 트위터 유저 사이에서 누가 해당 트윗을 작성했을 것인지를 예측할 수 있는 기능이다.

## 어플리케이션 개발 과정
---
### 개발에 사용된 모듈 및 패키지
- Tweepy : 트위터 API를 파이썬에서 활용가능하게 해주는 모듈.
- Flask : 파이썬으로 웹 어플리케이션을 개발하기 위한 프레임워크. API 어플리케이션을 만들기 위한 각종 편의 기능들을 제공한다.
- FLASK SQLAlchemy : ORM(Object Relational Mapper)의 한 종류로 Flask 프레임워크에서 데이터베이스와의 상호작용을 파이썬과 비슷한 객체형식으로 가능하게 해주는 모듈.


### 1. \_\_init\_\_.py & models.py 생성

- __init__.py : 어플리케이션을 initialize하기 위한 파일로 블루프린트를 사용하여 함수를 여러 곳으로 확장하고 분산하여 구현하였다. 블루프린트를 사용하지 않아도 웹서버 구현이 가능하지만 함수들을 분리하여 관리하면 어플리케이션에 기능이 많아지면 파일이 길어져 관리가 힘들어지기 때문에 분산관리의 이점이 드러나게 된다.
사용자가 웹페이지에서 url을 입력했을 때, 해당 url과 route 파일이 상호 연결될 수 있도록 해준다.
 
 - models.py : Flask_SQLAlchmy를 활용하여 DB에 데이터를 저장하는 형식을 지정하는 파일이다. 테이블, 컬럼 명 등을 지정하고 테이블간의 관계도 이곳에서 정해주게 된다.
  
### 2. 데이터베이스 연결

`__init__.py`와 `models.py` 파일을 작성했다면 어플리케이션에서 사용자가 입력한 정보를 저장할 데이터베이스를 구축하고 연결해야 한다.
데이터베이스는 간단히 코드 세줄로 구축 및 연결이 가능하다.



```py
# 데이터베이스 구축
FLASK_APP=twitter_app flask db init
# 데이터베이스 테이블 생성
FLASK_APP=twitter_app flask db migrate
#데이터베이스 테이블에 세부 컬럼 생성
FLASK_APP=twitter_app flask db upgrade

```


### 3. routes & templates 생성

`routes`폴더와 `templates`폴더는 각각의 파일들을 관리하게 되는데 `routes`파일에는 어플리케이션의 기능 구현에 대한 코드를 작성하게 되고 templates 폴더에 담겨지는 html 파일은 `routes` 파일로 구현된 기능을 웹페이지에 어떻게 뿌려줄지를 결정하는 역할을 한다.
쉽게 말하면 `routes`는 기능구현 파일, `html`파일은 웹에 어떻게 보여질지를 구성하는 파일이다. 

![Home](https://user-images.githubusercontent.com/70134676/102013133-8ec8ab00-3d91-11eb-90db-f038ff1e5a65.png)

- main_routes.py & index.html : Home 웹페이지에 접속하면 처음으로 보이는 페이지로 index.html로 User 테이블의 정보를 전달한다. 

![Add](https://user-images.githubusercontent.com/70134676/102013282-5b3a5080-3d92-11eb-8050-4d9dc4095b43.png)

- add_routes.py & add.html : = 사용자가 입력한 트위터 유저의 username을 add.html을 통하여 radd_routes.py로 전달하고 해당 유저의 정보와 트윗기록들을 User 테이블과 Tweet 테이블에  저장한다.

![Get](https://user-images.githubusercontent.com/70134676/102013390-35fa1200-3d93-11eb-85f0-2691ea5b5790.png)

- get_routes.py & get.html : 사용자가 입력한 트위터 유저의 username을 get.html을 통하여 get_routes.py로 전달하고 Tweet 테이블에 저장된 데이터중 해당 username 과 일치하는 레코드들을 쿼리하여 get.html 에 전달한다. get_html은 전달받은 레코드를 웹페이지에 출력한다.

![Delete](https://user-images.githubusercontent.com/70134676/102013503-f4b63200-3d93-11eb-8042-7e5f4c7bbd1c.png)

- delete_routes.py & delete.html :  사용자가 입력한 트위터 유저의 username을 delete.html을 통하여 delete_routes.py로 전달하고 해당 유저에 대한 User, Tweet 테이블 내의 정보를 모두 삭제한다.

![Update](https://user-images.githubusercontent.com/70134676/102013538-2deea200-3d94-11eb-9f70-cd9d68447f29.png)

- update_routes.py & update.html : add를 통해 저장된 데이터 중 사용자가 입력한 트위터 유저의 FullName을 delete.html을 통하여 delete_routes.py로 전달하고 User 테이블 내의 FullName을 업데이트 한다.

![Predict](https://user-images.githubusercontent.com/70134676/102013636-c5ec8b80-3d94-11eb-873a-d4835b9f2bb7.png)

- predict_routes.py = predict.html : 사용자가 입력한 트윗 내용에 대해 add를 통해 저장된 두 명의 트위터 유저 사이에서 누가 입력 트윗을 작성했을지 예측한다.
로지스틱회귀모델을 적용하여 데이터베이스에 저장된 유저의 트윗내용으로 학습하여 입력된 트윗내용을 예측하게 된다.

### 4. 어플리케이션 구동여부 확인 후 배포
어플리케이션을 본격적으로 웹에 배포하기 전에 로컬환경에서 앱의 구동여부를 확인해야 한다.
하지만 보통 개발과정에서 셀 수 없을 정도로 많은 에러를 접하게 되기 때문에 수시로 어플리케이션을 구동하여 각각의 파일들의 상호작용이 원활한지 확인하는 과정을 거치게 된다.
로컬에서 기능이 문제없이 작동한다면 웹에 배포하기 위하여 클라우드 플랫폼을 준비해야 한다.
나는 이번에 개발한 어플리케이션을 heroku라는 플랫폼을 활용하여 배포할 것이다.


---
## 코드스테이츠의 데이터사이언티스트 코스 세번째 섹션을 마무리하며..

항상 프로젝트를 진행하면서 이 프로젝트는 추후 현업에서 어떻게 사용할 수 있을까에 대한 고민을 하게 된다. 그 고민은 일주일 동안 열정을 쏟을 나의 프로젝트에 대해 더 애착을 가지게 해주고 나아가 더 나은 결과물을 얻을 수 있는 큰 동기부여가 된다.

이번 프로젝트는 데이터 사이언티스트의 주 업무라고 할 수 있는 데이터를 다루는 과정에서 친절하게 데이터가 주어지지 않았을 때를 대비한 훈련이라고 생각되었다. 공식적으로 제공하는 API를 활용하여 데이터베이스를 구축하고 축적된 데이터를 바탕으로 데이터 사이언티스트 본연의 임무를 가능하게 해주는 일종의 준비단계인 셈이다. 어떤 환경에서 일하게 될지 모르기 때문에 웹에서 직접 데이터베이스를 구축하고 데이터베이스를 활용하여 기능을 구현하는 연습은 좋은 데이터사이언티스트가 되는데도 큰 도움을 줄 것이다.

섹션에서 배웠던 모듈들만 활용한 제한적인 범위 내에서 수행된 프로젝트였지만 프로젝트를 진행하며 3주동안 배운 지식들이 퍼즐이 맞춰지는 듯한 신기한 경험과 더불어 각각의 패키지와 라이브러리들이 어떤 역할을 하는지 확실하게 이해할 수 있는 시간이었다. 더불어 에러메시지를 대하는 태도도 조금은 의연해진 것 같다. 하지만 여전히 나만의 코드를 독창적으로 작성할 수 없다는 한계점을 여지없이 드러내며 스스로에게 숙제를 안겨주기도 한 시원섭섭한 프로젝트였다.