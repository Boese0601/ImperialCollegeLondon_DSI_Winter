import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import io,exposure
from skimage.transform import rotate, AffineTransform, warp,rescale
from skimage.util import random_noise


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


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, batch_size=32, dim=(240, 240), n_channels=3,
                 n_classes=2, shuffle=True, seg=False):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.seg = seg
        self.on_epoch_end()
        self.hf=1
        self.vf=1
        self.rot_angle1=20
        self.rot_angle2=-20
        self.shift_x=0.1
        self.shift_y=0.1
        self.shear=0.2
        self.rescale=1.5
        self.noise=1
        self.blur=1

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

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if self.seg:
            y = np.empty((self.batch_size, *self.dim))
        else:
            y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # TODO: add data augmentation
            img = np.load(ID)
            if self.hf and random.random()<0.5:
                img=np.flip(img,axis=0)
            if self.vf and random.random()<0.5:
                img=np.flip(img,axis=1)
            if self.rot_angle1 and random.random()<0.5:
                img=rotate(img, angle=self.rot_angle1)
            if self.rot_angle2 and random.random()<0.5:
                img=rotate(img, angle=self.rot_angle2)
            if self.shift_x and random.random()<0.5:
                transform_x = AffineTransform(translation=(img.shape[0]*self.shift_x,0))  
                img = warp(img,transform_x) 
            if self.shift_y and random.random()<0.5:
                transform_y = AffineTransform(translation=(0,img.shape[1]*self.shift_y))  
                img = warp(img,transform_y)
            if self.shear and random.random()<0.5:
                shear_transform = AffineTransform(shear=self.shear)  
                img = warp(img,shear_transform)     
            if self.rescale and random.random()<0.5:
                scale_transform = AffineTransform(scale=(self.rescale,self.rescale))  
                img = warp(img,scale_transform)   
            if self.noise and random.random()<0.5:
                img=random_noise(img)
            if self.blur and random.random()<0.5:
                img=cv2.GaussianBlur(img, (11,11),0)
            X[i, ]=img

            # Store class
            if self.seg:
                y[i] = np.load(ID[:-8] + '_seg.npy')
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
    plot_samples('../data/Dataset')
