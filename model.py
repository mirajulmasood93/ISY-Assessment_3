import os   
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import batch_generator, INPUT_SHAPE
import argparse


np.random.seed(0)

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    """
    exploration of the dataset - we made changes in this model here
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    
    X = data_df[['forward', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model and we modified this model to  the ResNet neural network - we made changes in this model here
    """
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)  

    model = Sequential()
    model.add(resnet)

    model.add(Dropout(0.5))
      
    model.add(Flatten())
      
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    """
    Adam optimiser with the tunned learning rate - we made changes in this model here
    """
    optimizer = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        callbacks=[checkpoint],nb_val_samples=0.3,
                        verbose=1)
   
    
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='dataa')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=4)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.001)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)

if __name__ == '__main__':
    main()

