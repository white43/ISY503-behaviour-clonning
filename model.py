import csv
import os
import pathlib
import sys
from datetime import datetime as dt
from getopt import getopt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.losses import MSE
from keras.models import Sequential
from keras.optimizers import Adam
from pandas import Series

print(tf.__version__)

debug: bool = True

sources: list[str] = []
extra_steering: float = 0.2
validation_data_percent: float = 0.3
train_on_autonomous_center: bool = False

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
    "train-on-autonomous-center",
]

opts, args = getopt(sys.argv[1:], shortopts="", longopts=cli_longopts)

for opt, arg in opts:
    if opt == "debug":
        debug = True
    elif opt == "--sources":
        sources = arg.split(',')
    elif opt == "--train-on-autonomous-center":
        train_on_autonomous_center = True


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


def grayscale(img: Image) -> Image:
    grayed = ImageOps.grayscale(img)

    if debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_grayscale_origin.jpg")
        grayed.save("debug_augmentation_grayscale_processed.jpg")

    return grayed


def three_dimensional_grayscale(img: Image) -> Image:
    return Image.merge('RGB', (img, img, img))


def add_gray_layer_to_rgb_image(rgb: Image) -> np.ndarray:
    rgb_array = np.asarray(rgb)
    grayscale_array = np.asarray(grayscale(rgb))

    return np.concatenate((rgb_array, np.expand_dims(grayscale_array, axis=-1)), axis=2)


def equalize(img: Image) -> Image:
    equalized = ImageOps.equalize(img)

    if debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_equalize_origin.jpg")
        equalized.save("debug_augmentation_equalize_processed.jpg")

    return equalized


def save_autonomous_image(path: str, image: Image, steering: float) -> None:
    img_subdir: str = "/IMG/"
    write_mode = "a"

    if not os.path.exists(path):
        pathlib.Path(path + img_subdir).mkdir(parents=True, exist_ok=True)
        write_mode = "w"

    basename = dt.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg"
    image.save(path + img_subdir + basename)

    with open(path + "/driving_log.csv", write_mode) as fd:
        writer = csv.writer(fd)
        writer.writerow([img_subdir + basename, "", "", str(steering)])
        fd.close()


def is_autonomous_row(row: Series) -> bool:
    return pd.isna(row['right']) and pd.isna(row['left'])


def get_unit_of_data_from_autonomous_data(row: Series, steering: float) -> (Image, float):
    if is_autonomous_row(row):
        image = Image.open(row['center'])
    else:
        match np.random.choice(2):
            case 0:
                image = Image.open(row['left'])
                steering += extra_steering
            case 1:
                image = Image.open(row['right'])
                steering -= extra_steering
            case _:
                raise Exception("unexpected choice")

    return image, steering


def get_unit_of_data_from_human_gathered_data(row: Series, steering: float) -> (Image, float):
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

    return image, steering


def get_driving_logs(dirs: list[str]) -> pd.DataFrame:
    clear_data_list: list[pd.DataFrame] = []

    for dir in dirs:
        print("Reading " + dir, file=sys.stderr)

        csv = pd.read_csv(
            dir + '/driving_log.csv',
            delimiter=',',
            names=['center', 'left', 'right', 'steering'],
            usecols=[0, 1, 2, 3],
        )

        for column in ['center', 'left', 'right']:
            if csv[column].count() > 0:
                csv[column] = csv[column].map(lambda path: "%s/%s" % (dir, "/".join(path.split('/')[-2:])))

        clear_data_list.append(csv)

    return pd.concat(clear_data_list)


def get_datasets_from_logs(logs: pd.DataFrame, autonomous: bool) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []

    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []

    for index, row in logs.iterrows():
        steering = row['steering']

        if autonomous:
            image, steering = get_unit_of_data_from_autonomous_data(row, steering)
        else:
            image, steering = get_unit_of_data_from_human_gathered_data(row, steering)

        training_image = np.random.rand() > validation_data_percent

        if training_image:
            if np.random.rand() < 0.5:
                image = flip_horizontally(image)
                steering *= -1

            if np.random.rand() < 0.5:
                image = blur(image)

            if np.random.rand() < 0.5:
                image = grayscale(image)
                image = three_dimensional_grayscale(image)

        image = image.crop((
            crop_left,
            crop_top,
            origin_image_width - crop_right,
            origin_image_height - crop_bottom,
        ))

        image = equalize(image)

        # image = add_gray_layer_to_rgb_image(image)

        image = np.asarray(image)

        if training_image:
            train_x.append(image)
            train_y.append(steering)
        else:
            val_x.append(image)
            val_y.append(steering)

    return np.asarray(train_x), np.asarray(train_y), np.asarray(val_x), np.asarray(val_y)


def build_model() -> Sequential:
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(cropped_height(), cropped_width(), origin_colours)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())  # 1536 pixels
    model.add(Dropout(0.25))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss=MSE, optimizer=Adam(learning_rate=0.001))

    return model


def draw_plot(iterations, *args):
    for i in range(0, len(args)-1, 2):
        plt.plot(range(1, iterations + 1), args[i], label=args[i+1])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best', fontsize='small')

    plt.savefig("Loss history.jpg")


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
    model = build_model()
    print(model.summary())

    logs = get_driving_logs(sources)
    train_X, train_Y, val_X, val_Y = get_datasets_from_logs(logs, train_on_autonomous_center)
    history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=10, callbacks=model_callback_list())

    draw_plot(
        history.params['epochs'],
        history.history['val_loss'], 'Validation Loss',
        history.history['loss'], 'Training loss',
    )
