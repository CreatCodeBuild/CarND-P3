import numpy as np
import json

from keras.engine import Model
from keras.layers import Flatten, Dense, Convolution2D, Dropout
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from pandas import concat
from pandas import read_csv
import numpy as np
from scipy.misc import imread, imresize
from sklearn.utils import shuffle

h = 66
w = 200

def normalize(img):
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    return (img - mean) / std

def preprocess_image(img):
    img = imresize(img[60:150, :, :], (200, 66))
    return normalize(img)


def load_images(X):
    new = [0]*len(X)
    for i, path in enumerate(X):
        image = imread(path)
        image = preprocess_image(image)
        new[i] = image
    return new

def create_model(model_name):
    learning_rate=0.0001
    dropout=0.5
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3),
                            activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.save('models/'+model_name)


def train(model_name, batch_size, nb_epoch, valid_split, center_only, angle, sharp_turn):
    model = load_model('models/'+model_name)

    df = read_csv('data/driving_log.csv')
    if sharp_turn:
        df2 = read_csv('data/sharp_turn.csv')
        df = concat([df, df2])

    df = shuffle(df)
    df_validation = df[:int(len(df)*valid_split)]
    df = df[int(len(df)*valid_split):]

    X = []
    Y = []
    for c, l, r, s in zip(df['center'].values, df['left'].values, df['right'].values, df['steering'].values):
        X.append(c)
        Y.append(s)

        if not center_only:
            X.append(l)
            Y.append(s + angle)

            X.append(r)
            Y.append(s - angle)

    X, Y = shuffle(X, Y)
    X = load_images(X)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)

    valid_X = np.array(load_images(df_validation['center'].values))
    valid_Y = df_validation['steering'].values
    print(valid_X.shape, valid_Y.shape)

    hist = model.fit(X, Y, validation_data=(valid_X, valid_Y), shuffle=True, batch_size=batch_size, nb_epoch=nb_epoch)
    
    with open('models/'+model_name+'.json', 'w') as f:
        json.dump(hist.history, f)

    model.save('models/'+model_name)

if __name__ == '__main__':
    create_model('nvidia_center.h5')
    train('nvidia_center.h5', batch_size=512, nb_epoch=30, valid_split=0.1, center_only=True, angle=0.25, sharp_turn=False)

    create_model('nvidia_center_sharp_turn.h5')
    train('nvidia_center_sharp_turn.h5', batch_size=512, nb_epoch=30, valid_split=0.1, center_only=True, angle=0.25, sharp_turn=True)

    create_model('nvidia_cfr25_sharp_turn.h5')
    train('nvidia_cfr25_sharp_turn.h5', batch_size=512, nb_epoch=30, valid_split=0.1, center_only=False, angle=0.25, sharp_turn=True)

    create_model('nvidia_cfr25.h5')
    train('nvidia_cfr25.h5', batch_size=512, nb_epoch=30, valid_split=0.1, center_only=False, angle=0.25, sharp_turn=False)
