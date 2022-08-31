from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, RandomFlip, Dropout, InputLayer, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import keras
from helper import load_preprocess

BATCH_SIZE = 128

def get_model():
    model = Sequential([
        InputLayer(input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    train_x, train_y, test_x, test_y = load_preprocess()

    model = get_model()
    generator = ImageDataGenerator()
    train_iterator = generator.flow(train_x, train_y, batch_size=BATCH_SIZE)

    # fit model - write your code below
    steps = int(train_x.shape[0] / BATCH_SIZE)
    history = model.fit_generator(train_iterator, steps_per_epoch = steps, epochs = 100, validation_data = (test_x, test_y), )

train()
