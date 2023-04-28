# K-AI Manufacturing data PROJECT
* 중소벤처기업부, KAIST 주관의 ‘제2회 K-인공지능 제조데이터 분석 경진대회’ 참가 프로젝트로서,
  <br> 제조 분야에서 PLC 및 DBMS를 통해 수집된 데이터를 기반으로 프로젝트를 진행하였습니다.

<br/><br/>

# 프로젝트명: [ 이상탐지 기반 공정 품질 예측 모델 개발 ]
<br/>

## 프로젝트 배경 및 목적
* 분말 유크림 제조 공정 중 원료 전처리 작업의 가장 첫 번째 단계인 용해 공정 결과물의 균일함 여부가 <br/>
  최종 생산물의 품질을 결정함, 그런데 이 용해 공정 품질 관리에 어려움을 겪는 상황
  
<br/>
  
## 프로젝트 목표
* 용해 공정 데이터를 통해 완제품 품질 예측을 위한 모델 개발
* 제조업 분야의 이상 탐지(Anomaly Detection) 문제로 판단하여, 라벨 값으로 주어진 불량여부(TAG) 컬럼을 정답으로 두고 지도 학습을 시행
* 데이터셋에 나와 있는 변수들은 특히 영향력이 높은 변수인 만큼, 주어진 데이터셋에 나와 있는 주요 변수들은 모두 활용하여 품질을 예측하는 모델을 만들고자 하였음
* 제조업 산업 특성 상 라벨 컬럼에서 불량(NG)의 비율이 적은 불균형 데이터가 주어졌기 때문에, 학습을 균형 있게 진행하고자 K-nearest neighbors 기반의 SMOTE 기법을 이용하여 균등한 학습을 통해 모델을 구축하였음
  
<br/>

## 프로젝트 설명

<br/>

### 문제정의  
![002](https://user-images.githubusercontent.com/106140951/235034399-dafdc62f-7a4f-4bcf-ae6f-d5fd314ec0d6.jpg)

<br/>

![003](https://user-images.githubusercontent.com/106140951/235034650-360d1a38-b2fa-4a18-a94b-ea68e7fdddc4.jpg)

<br/>

### 제조데이터 정의 및 처리과정
![005](https://user-images.githubusercontent.com/106140951/235034439-92eaaed2-909f-4683-b169-4020b88916f8.jpg)

<br/>

![007](https://user-images.githubusercontent.com/106140951/235034699-1910874d-94e2-49fd-b34f-64637487f2a6.jpg)

<br/>

![010](https://user-images.githubusercontent.com/106140951/235034730-8d391eff-e7f0-4726-b503-6b8567712716.jpg)

<br/>

![014](https://user-images.githubusercontent.com/106140951/235034749-9e2148bc-4e0f-40e1-b0ac-934fd8866cbe.jpg)
#### SMOTE
* 일반적인 모델 학습에서는 훈련데이터와 테스트데이터의 라벨값의 비율을 맞춰줌, 편중된 데이터로 학습했을 경우 모델이 제대로 작동하기 어렵기 때문
* 그러나 제조업의 이상 탐지 분야에서는 라벨값의 임의 조정이 어려움, 따라서 본 프로젝트에서는 훈련데이터에서 균등하게 학습을 시키는 것에 초점을 맞추고 테스트데이터를 통해 모델이 잘 작동되는지 확인
<br/>

#### Window 정의
* RNN을 이용한 시계열데이터의 예측 문제에서 많이 사용되는 방법은 슬라이딩 윈도우(Sliding Window)를 사용하는 방법으로, 본 프로젝트에서도 이를 사용함
    - 인터벌(window size)은 10으로 잡았는데, 그 이유는 교반속도와 온도가 10을 주기로 일정한 주기성을 가지기 때문
* 과거의 feature vector 10개를 가지고 현재의 label y를 예측
    - 본 프로젝트에서 사용하는 RNN은 10개의 15차원 벡터를 입력값으로 받고, 마지막 timestep에서 하나의 실수값을 출력
    - 이 실수값은 다시 시그모이드 함수를 거쳐 0과 1사이의 값으로 변환
    - 이 예측값 y는 과거 10개의 timestep을 가지고 얻은, 특정 시각에서의 제품의 품질이 양품(label: 1)인 확률


<br/>

### 분석모델 개발
![015](https://user-images.githubusercontent.com/106140951/235034770-0b3e58b9-d840-40a2-9a1c-abc1310b8672.jpg)

* 순차 데이터(sequantial data)를 처리하기 위해 효과적인 모델로는 RNN이 있음, 그러나 RNN의 고질적인 문제인 장기 의존성 문제를 해결한 LSTM이 RNN의 대표적인 알고리즘으로 사용되는 편
* GRU는 LSTM을 간소화하여 만든 구조
    - LSTM에서 3개였던 게이트 수가 GRU에서 2개로 줄어들었고, LSTM에서 은닉상태벡터(hidden state vector)와 셀상태벡터(cell state vector) 모두를 사용했다면, GRU에서는 은닉상태벡터(hidden state vector)만을 사용
* 알고리즘 선정 이유
    - GRU는 LSTM보다 간단하게 구성되어 있지만, LSTM과 비슷한 성능을 내는 것으로 알려져 있음
    - 본 프로젝트에서 LSTM과 GRU 모두 사용해본 결과, GRU가 LSTM보다 조금 더 낫거나 비슷한 성능을 보이는 것으로 확인되었음
    - LSTM은 GRU보다 먼 과거를 효율적으로 포착하는 것으로 알려져 있는데, 본 프로젝트에서는 주요 특성(feature)들이 10 정도의 주기를 가지고 있어 아주 먼 과거를 포착하는 것이 큰 의미가 없을 것이라고 판단함
    - 복잡하지 않고 간단한 수열에 대한 문제에서는 GRU가 LSTM보다 더 나은 성능을 보인다는 연구가 있음
* GRU 레이어는 한 개만을 사용할 수도 있지만, 조금 더 정교한 모델 구조를 위해 여러 개의 GRU 레이어를 쌓기도 함
    - 본 프로젝트에서 1 layer GRU, 2 layer GRU, 3 layer GRU 등을 고려했지만 결과적으로 3 layer GRU가 가장 좋은 성능을 보였다
* - 활성화함수: RNN에서 표준적으로 쓰이는 하이퍼볼릭탄젠트(hyperbolic tangent) 함수
  - 옵티마이저: Adam

### 모델 구축 및 훈련
* 모델의 구성, 학습, 평가 등은 keras의 sequential API를 사용하여 구현
* 세 개의 GRU 레이어를 쌓고, 처음 두 레이어에서는 10개의 timestep 모두에서 은닉상태벡터가 출력되어야 하므로 return_sequences=True로 두었고, 마지막 레이어에서는 마지막 timestep에서만 결과가 출력되어야 하므로 return_sequences=False로 두었음
* 출력된 결과를 0과 1사이의 값으로 변환시켜주기 위해 시그모이드 함수를 활성화함수로 하는 선형 레이어를 하나 더 쌓아주었음

<br/>

### 분석결과 및 시사점
![017](https://user-images.githubusercontent.com/106140951/235034799-dcb26f56-018d-4c61-89ec-fc90bd1d2090.jpg)

<br/>

![019](https://user-images.githubusercontent.com/106140951/235034828-ef7a7568-da5b-4615-9be3-0fa0604a2aa4.jpg)

<br/>

### 회고 
* 학습했던 모델을 실제로 적용해볼 수 있는 경험이 되었다. </br>
* 지도학습이 아니라 비지도학습으로 접근했다면 어땠을까, 또 RNN 계열 이외의 다른 모델들도 적용해볼 수 있지 않았을까 하는 아쉬움이 있다.


