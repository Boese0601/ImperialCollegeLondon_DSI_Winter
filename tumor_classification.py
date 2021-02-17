import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tumor_head import load_data, DataGenerator, plot_history
from tensorflow.keras import layers
from tensorflow.keras.models import *
import tensorflow.keras.applications as app
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load data
print("[INFO] Loading data...")
train_list, val_list = load_data()


batch_size = 16
train_generator = DataGenerator(
    train_list,
    batch_size=batch_size
)
validation_generator = DataGenerator(val_list, batch_size=batch_size)
img_size = (240, 240)

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

# uncomment here if you want to finetune the top layer(classifier) of a pretrained network only.
# model.layers[0].trainable = False

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
