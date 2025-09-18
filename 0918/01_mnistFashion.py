import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# mnist의 fashion datasets 객체를 불러옴
fashion_mnist = tf.keras.datasets.fashion_mnist
# 데이터 로드해서 언패킹
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 데이터셋에 클래스 이름이 들어있지 않기 때문에, 별도의 변수로 저장
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Avkle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_images))
print(test_labels)

train_images = train_images /255.0
test_images = test_images / 255.0

'''
# 새 그래프를 그림
plt.figure()
# 이미지를 띄움
plt.imshow(train_images[0])
# 이미지의 스케일 표시
plt.colorbar()
# grid 옵션 False
plt.grid(False)
# 이미지 표시
plt.show()

# 10*10인치의 화면 공간을 확보
plt.figure(figsize=(10,10))
# 25개의 이미지를 띄울 거임
for i in range(25):
    # 가로, 세로, 인덱스 (1부터 시작)
    plt.subplot(5,5,i+1)
    # x,y의 공백을 없앰
    plt.xticks([])
    plt.yticks([])
    # grid 옵션 False
    plt.grid(False)
    # 이미지를 그림 (cmap : 그레이스케일로)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 라벨 추가
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
# 모델 생성
model = tf.keras.Sequential([
    # 28*28 numpy 배열을 1차원으로 flatten함
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(512, activation='relu'),
 	tf.keras.layers.Dense(32, activation='relu'),
	# 분류 문제니까 활성함수는 softmax
	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
	optimizer='adam',
 # 이 손실함수는 정답 데이터가 정수형이어야 함
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 # 분류 문제니까 학습 중에 관찰할 지표는 정확도
	metrics=['accuracy']
)

# 모델 개요 표시
model.summary()

# 모델 학습
model.fit(train_images,train_labels, epochs=10)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=2)
print("\nTest accuracy", test_acc)

# 모델 활용
predictions = model.predict(test_images)
print(predictions[0])

sum = 0
for p in predictions[0]:
    sum += p    
print(sum)

print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel(f"{format(class_names[predicted_label])} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})", color=color)
    
    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
'''
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
'''

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i, predictions[i], test_labels)
    
plt.tight_layout()
plt.show()