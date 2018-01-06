from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
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
    image = []
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = (cv2.imread(current_path))
        images.append(image)

        # create adjusted steering measurements for the side camera images
        measurement = float(line[3])
        correction = 0.2 # this is a parameter to tune
        if i == 1:
            measurement = measurement + correction
        elif i == 2:
            measurement = measurement - correction
        measurements.append(measurement)

"""
Data Augumentation Example
image_flipped = np.fliplr(image)
measurement_flipped = -measurement
"""

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()

"""
Normalization Example
pixel_normalized = pixel / 255
pixel_mean_centered = pixel_normalized - 0.5
"""

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

"""
Convolution2D explain
"""

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())

"""
Why 120
"""

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
