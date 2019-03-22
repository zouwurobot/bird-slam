import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import utils
import numpy as np


def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    velocity = Dense(output_dim)(x)
    direction = Dense(output_dim)(x)

    # coll = Dense(output_dim)(x)
    # coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[velocity, direction])
    print(model.summary())

    return model


def hard_mining_mse(k):
    def custom_mse(y_true, y_pred):
        hard_loss = keras.losses.mean_squared_error(y_true, y_pred)
        return hard_loss

    return custom_mse


def _main():
    output_dim = 2
    batch_size = 16
    experiment_rootdir = '/home/jing/PycharmProjects/dcnn/new_model'
    train_dir = '/home/jing/PycharmProjects/dcnn/training'
    val_dir = '/home/jing/PycharmProjects/dcnn/val'
    epochs = 100
    img_mode = 'rgb'
    img_channels=3
    img_width = 320
    img_height = 240
    crop_img_height = 200
    crop_img_width = 200
    initial_epoch= 0

    # Generate training data with real-time augmentation
    train_datagen = utils.DroneDataGenerator(rotation_range=0.2,
                                             rescale=1. / 255,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2)

    train_data_generator = train_datagen.flow_from_directory(train_dir,
                                                        shuffle=True,
                                                        color_mode=img_mode,
                                                        target_size=(img_width, img_height),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size=batch_size)

    # Generate validation data with real-time augmentation
    val_datagen = utils.DroneDataGenerator(rescale=1. / 255)

    val_data_generator = val_datagen.flow_from_directory(val_dir,
                                                    shuffle=True,
                                                    color_mode=img_mode,
                                                    target_size=(img_width, img_height),
                                                    crop_size=(crop_img_height, crop_img_width),
                                                    batch_size=FLAGS.batch_size)

    model = resnet8(img_width, img_height, img_channels, output_dim)
    # Initialize loss weights
    model.gama = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
    # Initialize number of samples for hard-mining
    model.k_mse_v = tf.Variable(batch_size, trainable=False, name='k_mse_v', dtype=tf.int32)
    model.k_mse_x = tf.Variable(batch_size, trainable=False, name='k_mse_x', dtype=tf.int32)
    optimizer = optimizers.Adam(decay=1e-5)

    # Configure training process
    model.compile(loss=[hard_mining_mse(model.k_mse_v),
                        hard_mining_mse(model.k_mse_x)],
                  optimizer=optimizer, loss_weights=[model.gama, 1])
    # Save model with the lowest validation loss
    weights_path = os.path.join(experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)
    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / batch_size))
    print('Training ------------')
    model.fit_generator(train_data_generator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=writeBestModel,
                        validation_data=val_data_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)

if __name__ == "__main__":
    _main()
