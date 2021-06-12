import os
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow import keras

BASE_DIR = BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_model = BASE_DIR + "/model"
dataset_dir = BASE_DIR + "/dataset"
data_dir = pathlib.Path(dataset_dir)

batch_size = 32
size = 180

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(size, size),
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(size, size),
    batch_size=batch_size
)

class_names = train_dataset.class_names

num_classes = len(class_names)

model = keras.models.load_model(data_model)

img_pred_dir = BASE_DIR + '/prediction/dandelion.jpg'

img_pred = keras.preprocessing.image.load_img(
    img_pred_dir, target_size=(size, size)
)

img_pred_array = keras.preprocessing.image.img_to_array(img_pred)
img_pred_array = tf.expand_dims(img_pred_array, 0)

prediction = model.predict(img_pred_array)
score = tf.nn.softmax(prediction[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
