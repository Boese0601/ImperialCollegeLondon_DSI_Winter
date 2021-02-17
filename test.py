import os
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tumor_head import load_data, DataGenerator, plot_history
from random import shuffle
from tensorflow.keras import layers
from tensorflow.keras.models import *
import tensorflow.keras.applications as app
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load data
print("[INFO] Loading data...")
path = '../data/Dataset/'
os.system('rmdir ..\\Train ..\\Val /s /q')
os.system('mkdir ..\\Train ..\\Val ..\\Train\\0 ..\\Train\\1 ..\\Val\\0 ..\\Val\\1')
plt.figure(figsize=(20, 20))
for CLASS in os.listdir(path):
    if not CLASS.startswith('.'):
        all_files = os.listdir(path + CLASS)
        files = [item for item in all_files if "img" in item]
        shuffle(files)
        img_num = len(files)
        for (n, file_name) in enumerate(files):
            name = os.path.join(path, CLASS, file_name)
            # 80% of images will be used for training, change the number here
            # to use different number of images for training your model.
            img = np.load(name).astype(int) * 255
            label = min(1, np.sum(np.load(str(name).replace('img', 'seg'))))
            file_name = str(file_name).replace('_img.npy', '.png')
            if n < 0.8 * img_num:
                cv2.imwrite(os.path.join('../Train/', str(label), file_name), img)
            else:
                cv2.imwrite(os.path.join('../Val/', str(label), file_name), img)

batch_size = 16
img_size = (240, 240)
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=app.vgg19.preprocess_input
)
validation_datagen = ImageDataGenerator(preprocessing_function=app.vgg19.preprocess_input)
train_generator = train_datagen.flow_from_directory(
    directory='../Train/',
    batch_size=batch_size,
    interpolation='bicubic',
    target_size=img_size,
)
validation_generator = validation_datagen.flow_from_directory(
    directory='../Val/',
    batch_size=batch_size,
    interpolation='bicubic',
    target_size=img_size,
)

# TODO: change the base model (originally VGG16)
base_model = app.vgg19.VGG19(
    weights=None,
    include_top=False,
    input_shape=img_size + (3,),
)

num_classes = 2
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4),
    metrics=['accuracy']
)
model.summary()

num_epochs = 30
earlystopping = EarlyStopping(
    monitor='accuracy',
    mode='max',
    patience=10
)

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
    callbacks=[earlystopping]
)

# plot
plot_history(history, batch_size)

# save model
model.save(f'res_cls/{num_epochs}_{batch_size}.h5')

# test model
"""test_dir = 'Test/'
model = load_model(f'res_cls/{num_epochs}_{batch_size}.h5')
test_list = []
for CLASS in os.listdir(test_dir):
    if not CLASS.startswith('.'):
        all_files = os.listdir(test_dir + CLASS)
        files = [item for item in all_files if "img" in item]
        for file_name in files:
            test_list.append(test_dir + CLASS + '/' + file_name)
test_generator = DataGenerator(test_list, batch_size=1)

predictions = []
y_test = []
for i in range(test_generator.__len__()):
    x_test, y = test_generator.__getitem__(i)
    y_test.append(y[0][1])
    prediction = model.predict(x_test)
    predictions.append(np.int(prediction[0][1]>0.5))
accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy = %.2f' % accuracy)"""
