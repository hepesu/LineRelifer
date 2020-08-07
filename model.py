import numpy as np
import datetime

from keras.models import Model
from keras.layers import Input, Activation, Reshape, BatchNormalization, MaxPool2D, Conv2D, Add, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception

SEED = 1
ITERATIONS = 10001
BATCH_SIZE = 8
IMG_SHAPE = (256, 256, 1)
IMG_HEIGHT, IMG_WIDTH, IMG_CHAN = IMG_SHAPE


def resnet_block(i, filters=64):
    x = Conv2D(filters, 3, padding='same')(i)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    return Activation('relu')(Add()([i, x]))


def build_generator():
    input_img = Input(IMG_SHAPE)

    x = Conv2D(64, 3, padding='same', activation='relu')(input_img)
    x = resnet_block(x)
    x = resnet_block(x)
    x = resnet_block(x)
    x = resnet_block(x)

    output_img = Conv2D(IMG_CHAN, 1, padding='same', activation='tanh')(x)

    return Model(input_img, output_img)


def build_discriminator():
    input_img = Input(IMG_SHAPE)

    x = Conv2D(16, 4, padding='same', strides=2, activation='relu')(input_img)
    x = Conv2D(32, 4, padding='same', strides=2, activation='relu')(x)
    x = Conv2D(64, 4, padding='same', strides=2, activation='relu')(x)
    x = Conv2D(128, 4, padding='same', strides=2, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, 4, padding='same', strides=2, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)

    validity = Dense(1, activation='sigmoid')(x)

    return Model(input_img, validity)


def get_image_batch(generator):
    img_batch = next(generator)

    if len(img_batch) != BATCH_SIZE:
        img_batch = next(generator)

    return img_batch


def train():
    # --------------------
    #  Model and Optimizer
    # --------------------
    optimizer = Adam(0.0002, 0.5)

    # Discriminator Model
    discriminator = build_discriminator()
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    discriminator.summary()

    # Generator Model
    generator = build_generator()
    generator.compile(
        loss='MSE',
        optimizer=optimizer
    )
    generator.summary()

    # Combined Model
    discriminator.trainable = False

    input_imgs = Input(IMG_SHAPE)
    refined_imgs = generator(input_imgs)
    valid = discriminator(refined_imgs)

    combined = Model(input_imgs, [refined_imgs, valid])
    combined.compile(
        loss=['MSE', 'binary_crossentropy'],
        loss_weights=[20, 1],
        optimizer=optimizer
    )

    # --------------------
    #  Data
    # --------------------
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.5,
        height_shift_range=0.5,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        preprocessing_function=xception.preprocess_input
    )

    line_generator = datagen.flow_from_directory(
        './data/line',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True
    )

    norm_generator = datagen.flow_from_directory(
        './data/norm',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True
    )

    real_validity = np.ones((BATCH_SIZE, 1))
    fake_validity = np.zeros((BATCH_SIZE, 1))

    start_time = datetime.datetime.now()
    for iteration in range(1, ITERATIONS + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        norm_batch = get_image_batch(norm_generator)
        line_batch = get_image_batch(line_generator)

        fake_batch = generator.predict_on_batch(norm_batch)

        disc_loss_real = discriminator.train_on_batch(line_batch, real_validity)
        disc_loss_fake = discriminator.train_on_batch(fake_batch, fake_validity)
        disc_loss = np.add(disc_loss_real, disc_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        norm_batch = get_image_batch(norm_generator)
        line_batch = get_image_batch(line_generator)

        combined_loss = combined.train_on_batch(norm_batch, [line_batch, real_validity])

        print('[Time %s] [Iteration %d] [D Loss: %f] [G Loss: %f,%f,%f]' % (
            datetime.datetime.now() - start_time,
            iteration,
            disc_loss[0],
            combined_loss[0],
            combined_loss[1],
            combined_loss[2]
        ))

        if iteration % 250 == 0:
            discriminator.save('./weight/%d_D.h5' % iteration)
            generator.save('./weight/%d_G.h5' % iteration)


if __name__ == "__main__":
    train()
