import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
 
def test_system(x):
    # 1차 함수
    return 0.4*x+8.0
 
def get_model():
    # 랜덤 시드 1000으로 고정.
    '''
    # 머신러닝 시 랜덤이 들어가는 과정이 있음
    - Dense의 w,b는 무작위 값으로 시작
    - shuffle=True일 때 epoch마다 데이터 순서가 무작위
    - 드롭아웃 (정규화기법) (이 코드에는 없음)
    # 난수 발생기의 시작값을 고정해서 실행할 때마다 같은 난수가 생성되도록 함
    - 재현성 : 내가 돌린 결과를 다른 사람이 그대로 재현할 수 있게 하기 위해    
    - 디버깅 : 매번 결과가 달라지면 버그인지, 랜덤성 때문인지 알기 어려움
    - 비교 실험 : 옵티마이저를 바꿔가며 성능 비교할 때, 초기 조건이 같아야 공정한 비교 가능
    '''
    tf.random.set_seed(1000)
    # 모델에 들어올 데이터의 형태를 정의하는 레이어
    '''
    shape=(1,) : 입력 데이터가 실수 1개짜리 스칼라라는 뜻
    name : 그냥 레이어에 이름 붙인 것. 디버깅/모델 구조 확인할 때 보기 편함
    '''
    input = Input(shape=(1,), name="Input")  
    # 완전연결 레이어 생성
    '''
    1 : 출력 유닛수 (여기서는 하나의 스칼라 값을 출력) (스칼라 : 하나의 숫자 값)
    activation='linear' : 활성화 함수 지정 (linear : 아무 변환 없이 선형합 그대로 출력)
    name : input과 마찬가지로 레이어 이름 지정
    (input) : Funtional API 방식에서 이 Dense레이어를 input 텐서에 적용. 결과가 output 텐서가 됨
    '''
    output = Dense(1, activation='linear', name="Output")(input)
    # 설정한 input과 output을 연결해 모델을 만듬
    # 입력, 출력을 여러 개 받을 수도 있기 때문에 리스트 형식으로 받음
    model = Model(inputs=[input], outputs=[output])
    # 옵티마이저 객체 생성 (손실함수를 최소화 하는데 사용)
    # learning rate : 학습률 (가중치를 얼마나 크게 업데이트 할 건지)
    # Adam : 경사하강법 기반의 최적화 알고리즘
    opt = keras.optimizers.Adam(learning_rate=0.0025)
    # 모델 학습 전에 손실함수, 최적화 방법, 평가 지표를 설정하는 단계
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    # 모델 구조를 콘솔에 출력
    model.summary()
    return model
 
if __name__ == '__main__':
    x_datas = np.array(range(-50, 51, 10)) # numpy 배열로 반환 (백터,행렬 연산 편리)
    y_datas = []
    for x in x_datas:
        y_datas.append(test_system(x))
    y_datas = np.array(y_datas)
 
    # plt.scatter(x_datas, y_datas)
    # plt.show()
    model = get_model()
 
    # model.fit : 모델 학습을 시작하는 함수
    '''
    - 입력값, 정답값을 주면, 모델의 가중치를 조정하면서 loss를 최소화하도록 학습함
    # fit 함수는 history 객체를 반환함 (학습 과정에서 발생하는 모든 기록과 상태를 담고 있음)
    - history.history : 학습 과정에서 기록된 손실값과 평가 지표가 들어있음
    # 즉 모델 학습 + 학습 로그 저장이 동시에 이루어짐
    '''
    history = model.fit(x_datas, y_datas, epochs=8000, shuffle=True)
    # 선 그래프를 그리는 함수
    '''
    history.history['loss'] : epoch마다 기록된 손실값 리스트
    'b' : 그래프 선 색깔 지정 (b : blue)
    label='loss' : 범례(legend)에 표시될 이름
    '''
    plt.plot(history.history['loss'], 'b', label='loss')
    plt.show()
 
    x_test = np.array(range(-45, 45, 10))
    # 학습된 모델을 이용해서 입력데이터에 대한 예측값을 계산하는 함수
    result = model.predict(x_test)
 
    # 산점도를 그리는 함수
    '''
    x,y 좌표
    c : 점의 색깔
    s : 점의 크기
    '''
    plt.scatter(x_datas, y_datas, c='blue', s=5)
    plt.scatter(x_test, result, c='red', s=5)
    plt.show()
 
    # 모델의 파라미터 값 numpy 배열로 반환 후 출력
    weights = model.get_weights()
    print("W : ", weights[0], "   b : ", weights[1])