from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import shutil
import tensorflow as tf
from tumor_head import load_data, DataGenerator, plot_history
from tumor_seg_model import unet, unet_plus_plus, unet_3p
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load data
print("[INFO] Loading data...")
train_list, val_list = load_data(seg=True)

batch_size = 8
train_generator = DataGenerator(
    train_list,
    batch_size=batch_size,
    horizontal_flip=True,
    vertical_flip=True,
    seg=True
)
validation_generator = DataGenerator(val_list, batch_size=batch_size, seg=True)
IMG_SIZE = (240, 240)


# TODO: change model (originally unet)
model = unet_plus_plus()
model.summary()

# train model
num_epochs = 25
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
)

# plot
plot_history(history, batch_size)

# save model
model.save(f'res_seg/{num_epochs}_{batch_size}.h5')
