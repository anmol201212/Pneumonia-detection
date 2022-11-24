from json import load
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import cv2

def load_data(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (180, 180))/255
    # expand dims
    img = np.expand_dims(img, axis = 0)
    return img

def make_model():
    image_size = (180,180,3)
    def conv_block(filters):
        block = tf.keras.Sequential([
                tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
                tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D()
            ]
        )

        return block

    def dense_block(units, dropout_rate):
        block = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        return block
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(image_size[0], image_size[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('xray_model.h5')
    
    return model

model = make_model()

def load_output(ROOT):
    data = load_data(ROOT + '/static/upload/1.jpg')
    pred = model(data)
    pred = float(pred)
    if pred > 0.5:
        return 1
    return 0
    # print(int(np.argmax(pred, axis = 1)))  
    # return int(pred)

# print(load_output())