import sys
from datetime import datetime as dt
from getopt import getopt

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from keras.losses import MSE
from keras.models import Sequential
from keras.optimizers import Adam

print(tf.__version__)

debug: bool = True

sources: list[str] = []
extra_steering: float = 0.15
validation_data_percent: float = 0.3

origin_image_width: int = 320
origin_image_height: int = 160
origin_colours: int = 3
crop_left: int = 0
crop_top: int = 55
crop_right: int = 0
crop_bottom: int = 25

cli_longopts: list[str] = [
    "debug",
    "sources=",
]

opts, args = getopt(sys.argv[1:], shortopts="", longopts=cli_longopts)

for opt, arg in opts:
    if opt == "debug":
        debug = True
    elif opt == "--sources":
        sources = arg.split(',')


def crop(img: Image) -> Image:
    return img.crop((
        crop_left,
        crop_top,
        origin_image_width - crop_right,
        origin_image_height - crop_bottom,
    ))


def cropped_height() -> int:
    return origin_image_height - crop_top - crop_bottom


def cropped_width() -> int:
    return origin_image_width - crop_left - crop_right


def flip_horizontally(img: Image) -> Image:
    flipped = ImageOps.mirror(img)

    if debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_flip_origin.jpg")
        flipped.save("debug_augmentation_flip_processed.jpg")

    return flipped


def blur(img: Image) -> Image:
    blurred = img.filter(ImageFilter.GaussianBlur(3))

    if debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_blur_origin.jpg")
        blurred.save("debug_augmentation_blur_processed.jpg")

    return blurred


def get_driving_logs() -> pd.DataFrame:
    clear_data_list: list[pd.DataFrame] = []

    for source in sources:
        print("Reading " + source, file=sys.stderr)

        csv = pd.read_csv(
            source + '/driving_log.csv',
            delimiter=',',
            names=['center', 'left', 'right', 'steering'],
            usecols=[0, 1, 2, 3],
        )

        for column in ['center', 'left', 'right']:
            csv[column] = csv[column].map(lambda path: "%s/%s" % (source, "/".join(path.split('/')[-2:])))

        clear_data_list.append(csv)

    return pd.concat(clear_data_list)


def get_datasets_from_logs(logs: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []

    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []

    for index, row in logs.iterrows():
        steering = row['steering']

        match np.random.choice(3):
            case 0:
                image = Image.open(row['center'])
            case 1:
                image = Image.open(row['left'])
                steering += extra_steering
            case 2:
                image = Image.open(row['right'])
                steering -= extra_steering
            case _:
                raise Exception("unexpected choice")

        training_image = np.random.rand() > validation_data_percent

        if training_image:
            if np.random.rand() < 0.5:
                image = flip_horizontally(image)
                steering *= -1

            if np.random.rand() < 0.5:
                image = blur(image)

        image = image.crop((
            crop_left,
            crop_top,
            origin_image_width - crop_right,
            origin_image_height - crop_bottom,
        ))

        if training_image:
            train_x.append(np.asarray(image))
            train_y.append(steering)
        else:
            val_x.append(np.asarray(image))
            val_y.append(steering)

    return np.asarray(train_x), np.asarray(train_y), np.asarray(val_x), np.asarray(val_y)


def build_model() -> Sequential:
    model = Sequential()

    model.add(Lambda(lambda x: x / 255, input_shape=(cropped_height(), cropped_width(), origin_colours)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())  # 512 pixels
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss=MSE, optimizer=Adam())

    return model


def model_callback_list() -> list[Callback]:
    list: list[Callback] = []

    list.append(
        ModelCheckpoint(
            "model-{}.h5".format(dt.now().strftime("%Y-%m-%d-%H-%M-%S")),
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        ),
    )

    return list


if __name__ == '__main__':
    logs = get_driving_logs()
    train_X, train_Y, val_X, val_Y = get_datasets_from_logs(logs)
    model = build_model()
    model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=5, callbacks=model_callback_list())
