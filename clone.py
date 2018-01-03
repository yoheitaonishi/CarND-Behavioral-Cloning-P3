from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import csv
import cv2
import numpy as np
from IPython import embed
from IPython.terminal.embed import InteractiveShellEmbed

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
del lines[0] # delete csv header

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
'''
pixel_normalized = pixel / 255
pixel_mean_centered = pixel_normalized - 0.5
'''
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
'''
Convolution2D explain
'''
model.add(Convolution2D( 6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D( 6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
'''
Why 120
'''
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

'''
6428/6428 [==============================] - 304s - loss: 0.6888 - val_loss: 0.0141
Epoch 2/7
6428/6428 [==============================] - 244s - loss: 0.0119 - val_loss: 0.0126
Epoch 3/7
6428/6428 [==============================] - 243s - loss: 0.0106 - val_loss: 0.0116
Epoch 4/7
6428/6428 [==============================] - 846s - loss: 0.0097 - val_loss: 0.0114
Epoch 5/7
6428/6428 [==============================] - 243s - loss: 0.0089 - val_loss: 0.0109
Epoch 6/7
6428/6428 [==============================] - 243s - loss: 0.0080 - val_loss: 0.0110
Epoch 7/7
6428/6428 [==============================] - 1035s - loss: 0.0071 - val_loss: 0.0110
'''
