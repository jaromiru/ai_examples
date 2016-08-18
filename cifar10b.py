# CIFAR10 recognition implementation with Keras
# Convolutional NN
#
# only with Y (luminance) channel
# C16x5x5 > MaxPool > C32x4x4 > C32x4x4 > D256 > D10
# ----------------------------------------------------

import os    
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"   

from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import cifar10

import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train_RGB = X_train.astype('float32') / 255
X_test_RGB = X_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# convert to YUV (only need luminance)
def toLuminosity( img ):		
	o = np.add(0.299 * img[0], 0.587 * img[1]);
	o = np.add(o, 0.114 * img[2])	
	return o.reshape(1, 32, 32)

print('Converting dataset to luminosity...')
X_train = np.zeros( (X_train_RGB.shape[0], 1, 32, 32) )
for i, o in enumerate(X_train_RGB):
	X_train[i] = toLuminosity(o)

X_test = np.zeros( (X_test_RGB.shape[0], 1, 32, 32) )
for i, o in enumerate(X_test_RGB):
	X_test[i] = toLuminosity(o)

model = Sequential()

model.add(Convolution2D(16, 5, 5, border_mode="same", input_shape=(1, 32, 32)))
model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Convolution2D(32, 4, 4, border_mode="same"))
model.add(Activation("relu"))

model.add(Convolution2D(32, 4, 4, border_mode="same"))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(output_dim=256))
model.add(Activation("relu"))

model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Starting training...')
for i in range(100):
	model.fit(X_train, y_train, nb_epoch=1, batch_size=50)

	loss_and_metrics = model.evaluate(X_test, y_test, batch_size=500)
	print(loss_and_metrics)

	model.save('model_cifar10b.h5')