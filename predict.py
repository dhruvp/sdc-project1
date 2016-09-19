#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import csv
from scipy import misc
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import pdb

# from server import client_generator
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def get_model(time_len=1):
    input_shape = (80, 320, 3)  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
            output_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
    # filename = './steering_angle.json'
    # with open(filename, 'r') as jfile:
    #     model = model_from_json(json.load(jfile))
    #     model.compile("adam", "mse")
    #     weights_file = filename.replace('json', 'keras')
    #     model.load_weights(weights_file)
    #     return model


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Steering angle model trainer')
    # parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    # parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    # parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
    # parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    # parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    # parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    # parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
    # parser.set_defaults(skipvalidate=False)
    # parser.set_defaults(loadweights=False)
    # args = parser.parse_args()

    y = []
    X = []
    with open('driving_log.csv', 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for row in reader:
            img_center_filename = row[0]
            img_left_filename = row[1]
            img_right_filename = row[2]

            img_center = (misc.imread('./' + img_center_filename))[80:, :, :]
            img_left = (misc.imread('./' + img_left_filename))[80:, :, :]
            img_right = (misc.imread('./' + img_right_filename))[80:, :, :]
            steering_angle = float(row[3])

            num_samples = 1
            left_steering_angle = steering_angle
            right_steering_angle = steering_angle

            if (steering_angle > 0):
                num_samples = 10

            for i in range(num_samples):
                X.append(img_center)
                y.append(steering_angle)
                X.append(img_left)
                y.append(left_steering_angle)
                X.append(img_right)
                y.append(right_steering_angle)

        X = np.asarray(X)
        y = np.asarray(y)

    model = get_model()

    # model.fit(X, y, nb_epoch=20)
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.15,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        featurewise_std_normalization=False,
        featurewise_center=False
    )

    model.fit_generator(
        datagen.flow(X, y, batch_size=32), samples_per_epoch=len(X), nb_epoch=10) 

    model.save_weights('first_try.keras')
    with open('./first_try.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    # model.fit_generator(
    #   train_generator,
    #   samples_per_epoch=2000,
    #   nb_epoch=50,
    #   validation_data=validation_generator,
    #   nb_val_samples=800
    # )

    # model.save_weights('first_try.keras')

    # model.fit(x_train, y_train, validation_data=(x_val, y_val))

    # model.fit_generator(
    #     gen(20, args.host, port=args.port),
    #     samples_per_epoch=10000,
    #     nb_epoch=args.epoch,
    #     validation_data=gen(20, args.host, port=args.val_port),
    #     nb_val_samples=1000
    # )
    # print("Saving model weights and configuration file.")

    # if not os.path.exists("./outputs/steering_model"):
    #         os.makedirs("./outputs/steering_model")

    # model.save_weights("./outputs/steering_model/steering_angle.keras", True)
    # with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    #     json.dump(model.to_json(), outfile)