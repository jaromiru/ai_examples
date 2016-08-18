# CIFAR10 recognition implementation with Keras
# Convolutional NN
# ----------------------------------------------------

import os    
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"   

from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode="same", input_shape=(3, 32, 32)))
model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Convolution2D(16, 3, 3, border_mode="same"))
model.add(Activation("relu"))

model.add(Convolution2D(16, 3, 3, border_mode="same"))
model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Convolution2D(8, 2, 2, border_mode="same"))
model.add(Activation("relu"))

model.add(Convolution2D(8, 2, 2, border_mode="same"))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(output_dim=256))
model.add(Activation("relu"))

model.add(Dense(output_dim=128))
model.add(Activation("relu"))

model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(100):
	model.fit(X_train, y_train, nb_epoch=1, batch_size=25)

	loss_and_metrics = model.evaluate(X_test, y_test, batch_size=500)
	print(loss_and_metrics)

	model.save('model_cifar10.h5')