import os # 파일 핸들링
from PIL import Image # 이미지 관련 라이브러리
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input


# traning data 생성

# 디렉터리의 이름을 리스트로 갖고 옴
all_files = [] # 파일 이름을 모을 리스트
for i in range(0, 10):
	path_dir = './images/training/{0}'.format(i) # training 폴더를 경로로 지정 	
	file_list = os.listdir(path_dir) # training 폴더 내 정보들을 리스트화
	file_list.sort() # 오름차순으로 정렬	
	all_files.append(file_list) # 하나로 모음
	# print(all_files) # 리스트 출력
 

x_train_datas = []
y_train_datas = []
for num in range(0,10):	# 폴더 하나씩 순회
	for numbers in all_files[num]:	# 폴더 안의 파일들 하나씩 순회
		img_path = f'./images/training/{num}/{numbers}' # 이미지 경로 저장
		print(f"load : {img_path}") # 이미지 경로 출력
		img = Image.open(img_path) # img 객체 저장
		imgarr = np.array(img) / 255.0	# 이미지 전처리(계산에 용이하도록 숫자의 범위를 줄임)
		x_train_datas.append(np.reshape(imgarr, newshape=(784,1))) # img객체를 numpy 배열로 변환하여 x_train_datas에 저장
		y_tmp = np.zeros(shape=(10))
		y_tmp[num] = 1
		y_train_datas.append(y_tmp) # num을 y_train_datas에 저장
  
print(len(x_train_datas))
print(len(y_train_datas))
	
 
# testing data 생성

# 디렉터리의 이름을 리스트로 갖고 옴
all_files = [] # 파일 이름을 모을 리스트
for i in range(0, 10):
	path_dir = './images/testing/{0}'.format(i) # testing 폴더를 경로로 지정 	
	file_list = os.listdir(path_dir) # testing 폴더 내 정보들을 리스트화
	file_list.sort() # 오름차순으로 정렬	
	all_files.append(file_list) # 하나로 모음
	# print(all_files) # 리스트 출력
 

x_test_datas = []
y_test_datas = []
for num in range(0,10):	# 폴더 하나씩 순회
	for numbers in all_files[num]:	# 폴더 안의 파일들 하나씩 순회
		img_path = f'./images/testing/{num}/{numbers}' # 이미지 경로 저장
		print(f"load : {img_path}") # 이미지 경로 출력
		img = Image.open(img_path) # img 객체 저장
		imgarr = np.array(img) / 255.0
		x_test_datas.append(np.reshape(imgarr, newshape=(784,1))) # img객체를 numpy 배열로 변환하여 x_test_datas에 저장
		y_tmp = np.zeros(shape=(10))
		y_tmp[num] = 1
		y_test_datas.append(y_tmp) # num을 y_test_datas에 저장

print(len(x_test_datas))
print(len(y_test_datas))

x_train_datas = np.reshape(x_train_datas, newshape=(-1,784))
y_train_datas = np.reshape(y_train_datas, newshape=(-1,10))
x_test_datas = np.reshape(x_test_datas, newshape=(-1,784))
y_test_datas = np.reshape(y_test_datas, newshape=(-1,10))

input = Input(shape=(784,), name="Input")
hidden = layers.Dense(512, activation="relu", name = "Hidden1")(input)
output = layers.Dense(10, activation="softmax", name="Output")(hidden)

model = keras.Model(inputs=[input], outputs=[output])
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

history = model.fit(x_train_datas, y_train_datas, epochs=5, shuffle=True,
                    validation_data=(x_test_datas, y_test_datas))

plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_accuracy'], 'r')
plt.show()

test_liss, test_acc = model.evaluate(x_test_datas, y_test_datas)
print("테스트 정확도 :", test_acc)

'''
num = 3
target = 100
# 이미지파일을 열어서 이미지 객체를 반환하여 img 변수에 대입
img = Image.open('./images/training/{}/'.format(num) + all_files[num][target])
# img = Image.open(f'./images/training/{num}/{all_files[num][target]}') # 이 방식도 가능 
img_arr = np.array(img) # 이미지 정보를 numpy 배열로 변환
print(img_arr) # numpy 배열 출력
print(img_arr.shape) # numpy 배열의 형식
'''
'''
plt.imshow(img) # 이미지를 그림
plt.show() # 그린 이미지를 화면에 표시
'''