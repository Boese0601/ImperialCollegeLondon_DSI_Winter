import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image.affine_transformations import *


def load_data(path='../data/Dataset/', seg=False):
    train_list = []
    val_list = []
    classes = ['Yes'] if seg else ['No', 'Yes']
    for CLASS in classes:
        if not CLASS.startswith('.'):
            all_files = os.listdir(path + CLASS)
            files = [item for item in all_files if "img" in item]
            random.shuffle(files)
            img_num = len(files)
            for (n, file_name) in enumerate(files):
                # 80% of images will be used for training, change the number here
                # to use different number of images for training your model.
                if n < 0.8 * img_num:
                    train_list.append(os.path.join(path, CLASS, file_name))
                else:
                    val_list.append(os.path.join(path, CLASS, file_name))

    return train_list, val_list


def transform(x, transform_parameters=None):
    x = apply_affine_transform(x,
                               transform_parameters.get('theta', 0),
                               transform_parameters.get('tx', 0),
                               transform_parameters.get('ty', 0),
                               transform_parameters.get('shear', 0),
                               transform_parameters.get('zx', 1),
                               transform_parameters.get('zy', 1),
                               row_axis=0,
                               col_axis=1,
                               channel_axis=2)
    if transform_parameters.get('channel_shift_intensity') is not None:
        x = apply_channel_shift(x,
                                transform_parameters['channel_shift_intensity'],
                                2)
    if transform_parameters.get('flip_horizontal', False):
        x = flip_axis(x, 1)
    if transform_parameters.get('flip_vertical', False):
        x = flip_axis(x, 0)
    return x


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, batch_size=32, dim=(240, 240), n_channels=3,
                 n_classes=2, shuffle=True, seg=False, channel_shift_range=0.0,
                 height_shift_range=0.0, width_shift_range=0.0, rescale=None,
                 horizontal_flip=False, vertical_flip=False, rotation_range=0,
                 shear_range=0.0, zoom_range=0.0):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.seg = seg
        self.channel_shift_range = channel_shift_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.rescale = rescale
        self.hf = horizontal_flip
        self.vf = vertical_flip
        self.rr = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif (len(zoom_range) == 2 and
              all(isinstance(val, (float, int)) for val in zoom_range)):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_random_transform(self, seed=None):
        """Generates random parameters for a transformation. Copied from ImageDataGenerator"""
        img_row_axis = 0
        img_col_axis = 1

        if seed is not None:
            np.random.seed(seed)

        if self.rr:
            theta = np.random.uniform(-self.rr, self.rr)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= self.dim[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= self.dim[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2)

        flip_horizontal = (np.random.random() < 0.5) * self.hf
        flip_vertical = (np.random.random() < 0.5) * self.vf

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                        self.channel_shift_range)

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity}

        return transform_parameters

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if self.seg:
            y = np.empty((self.batch_size, *self.dim))
        else:
            y = np.empty((self.batch_size,), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = np.load(ID)
            # data augmentation
            params = self.get_random_transform()
            X[i,] = transform(img, params)

            # Store class
            if self.seg:
                label = np.expand_dims(np.load(ID[:-8] + '_seg.npy'), -1)
                y[i] = np.squeeze(transform(label, params))
            else:
                y[i] = min(1, np.sum(np.load(ID.split('_')[0] + '_seg.npy')))

        if not self.seg:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y


def plot_samples(img_path, n=20):
    files_list = []
    labels_list = []
    for path, subdirs, all_files in os.walk(img_path):
        files = [item for item in all_files if "img" in item]
        for name in files:
            files_list.append(os.path.join(path, name))
            labels_list.append(path.split('/')[1])
    imgs_lbls = list(zip(files_list, labels_list))
    random.shuffle(imgs_lbls)
    files_list, labels_list = zip(*imgs_lbls)
    j = 5
    i = int(n / j)
    plt.figure(figsize=(15, 10))
    k = 1
    for file, lbl in zip(files_list[:n], labels_list[:n]):
        img = np.load(file)
        plt.subplot(i, j, k)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.xlabel(lbl)
        k += 1
    plt.tight_layout()
    plt.show()


def plot_history(history, batch_size):
    (task, label) = ('seg', 'Dice score') if 'dice_score' in history.history.keys() else ('cls', 'Accuracy')
    num_epochs = len(history.history['loss'])
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    except KeyError:
        acc = history.history['dice_score']
        val_acc = history.history['val_dice_score']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(history.epoch) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Set')
    plt.plot(epochs_range, val_acc, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(f'Model {label}')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Set')
    plt.plot(epochs_range, val_loss, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.savefig(f'res_{task}/{num_epochs}_{batch_size}.png')


if __name__ == '__main__':
    label = np.load('../data/Dataset/Yes/1_img.npy')
    print(label.shape)
    label = np.expand_dims(label, -1)
    print(label.shape)
    label = np.squeeze(label)
    print(label.shape)
