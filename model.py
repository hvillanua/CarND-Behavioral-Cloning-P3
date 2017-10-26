
# coding: utf-8

import os
import csv

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Cropping2D
from sklearn.utils import shuffle

# Check image shape
images_path = './data/IMG/'
img_ex = cv2.imread(images_path + os.listdir(images_path)[0])
input_shape = img_ex.shape
print('Image shape:', img_ex.shape)

# Read training csv
lines = []
data_path = './data/driving_log.csv'
with open(data_path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

lines = np.array(lines)
print(lines.shape)
plt.hist(lines[:,3].astype(float))


# Define helper functions
def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def _preprocess(img):
    yuv = bgr2yuv(img)
    return yuv

def preprocess(images):
    return np.array([_preprocess(img) for img in images])

def augment_data(features, labels):
    augmented_imgs, augmented_steerings = [], []
    for img, steer in zip(features, labels):
        if abs(steer) > 0.3:
            #only augment turns with steering > 0.3
            augmented_imgs.append(cv2.flip(img, 1))
            augmented_steerings.append(-steer)
        
    augmented_imgs = np.concatenate([features, np.array(augmented_imgs)])
    augmented_steerings = np.concatenate([labels, np.array(augmented_steerings)])
    return augmented_imgs, augmented_steerings

def generator(samples, batch_size=64):
    num_samples = len(samples)
    correction = 0.15
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # remove some of the straight driving, otherwise the network will not learn to turn properly
                if abs(steering_center) > 0.005 or (abs(steering_center) < 0.005 and np.random.random() > 0.5):
                     steering_left = steering_center + correction
                     steering_right = steering_center - correction
                     img_center = cv2.imread(images_path + batch_sample[0].split('/')[-1])
                     img_left = cv2.imread(images_path + batch_sample[1].split('/')[-1])
                     img_right = cv2.imread(images_path + batch_sample[2].split('/')[-1])
                     images.extend([img_center, img_left, img_right])
                     steering.extend([steering_center, steering_left, steering_right])

            x_train = np.array(images)
            y_train = np.array(steering)
            x_train = preprocess(x_train)
            x_train, y_train = augment_data(x_train, y_train)
            yield shuffle(x_train, y_train)


# Create model (Nvidia's)
batch_size = 256
epochs = 5
num_val = 500
shuffled = shuffle(lines)
train_generator = generator(shuffled[:-num_val], batch_size)
validation_generator = generator(shuffled[-num_val:], batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((70, 25),(0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam')


# Number of total samples is number of original images * 3 images 
# (center, right, left) * 2 (original, flipped)
model.fit_generator(train_generator, samples_per_epoch=(len(lines)-num_val)*3*2,
                    validation_data=validation_generator,
                    nb_val_samples=num_val*3*2,
                    nb_epoch=epochs)

save_model_path = './model/'
if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
model.save(save_model_path + 'model.h5')

