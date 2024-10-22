import argparse
import csv
import math
import os
import pathlib
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.layers import Conv2D, Dense, Dropout, Flatten, Input, Rescaling
from keras.api.losses import MeanSquaredError
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.src.callbacks import TensorBoard
from pandas import Series

print(tf.__version__)

origin_image_width: int = 320
origin_image_height: int = 160
origin_colours: int = 3
crop_left: int = 0
crop_top: int = 55
crop_right: int = 0
crop_bottom: int = 25


def crop(img: Image) -> Image:
    """ This function cuts off irrelevant for steering angle predictions information (sky and hood). The following
    variables will be used to calculate area to leave: origin_image_width, origin_image_height, crop_left, crop_top,
    crop_right, crop_bottom.

    Parameters
    ----------
    img : Image
        PIL.Image object to perform cropping.

    Returns
    -------
    Image
        Cropped PIL.Image object
    """
    return img.crop((
        crop_left,
        crop_top,
        origin_image_width - crop_right,
        origin_image_height - crop_bottom,
    ))


def cropped_height() -> int:
    """ This function calculates the height of the input shape to pass it to the input layer.

    Returns
    -------
    int
        The height of the input shape after cropping
    """
    return origin_image_height - crop_top - crop_bottom


def cropped_width() -> int:
    """ This function calculates the width of the input shape to pass to the input layer.

    Returns
    -------
    int
        The height of the input shape after cropping
    """
    return origin_image_width - crop_left - crop_right


def flip_horizontally(img: Image) -> Image:
    """ This function is used for dataset augmentation. It flips input image horizontally.

    Parameters
    ----------
    img : Image
        Original PIL.Image object to perform flip operation

    Returns
    -------
    Image
        Flipped PIL.Image object
    """
    flipped = ImageOps.mirror(img)

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_flip_origin.jpg")
        flipped.save("debug_augmentation_flip_processed.jpg")

    return flipped


def blur(img: Image) -> Image:
    """ This function is used for dataset augmentation. It blurs input image with the Gaussian Blur algorithm. Is not
    used by default in the current implementation, but can be easily activated if needed in `model.get_datasets_from_logs`

    Parameters
    ----------
    img : Image
        Original PIL.Image object to perform blur operation

    Returns
    -------
    Image
        Blurred PIL.Image object
    """
    blurred = img.filter(ImageFilter.GaussianBlur(3))

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_blur_origin.jpg")
        blurred.save("debug_augmentation_blur_processed.jpg")

    return blurred


def grayscale(img: Image) -> Image:
    """ This function is used for dataset augmentation. It converts input images to grayscale. Is not used by default in
    the current implementation, but can be easily activated if needed in `model.get_datasets_from_logs`

    Parameters
    ----------
    img : Image
        Original PIL.Image object to perform grayscale operation

    Returns
    -------
    Image
        Converted to grayscale PIL.Image object
    """
    grayed = ImageOps.grayscale(img)

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_grayscale_origin.jpg")
        grayed.save("debug_augmentation_grayscale_processed.jpg")

    return grayed


def three_dimensional_grayscale(img: Image) -> Image:
    """ This function is used to create a 3-channel grayscale image from the 1-channel grayscale image to use it along
    with real RGB images.

    Parameters
    ----------
    img : Image
        1-channel grayscale PIL.Image object to convert to 3-channel grayscale object

    Returns
    -------
    Image
        3-channel grayscale PIL.Image object
    """
    return Image.merge('RGB', (img, img, img))


def equalize(img: Image) -> Image:
    """ This function is used to equalize (normalize) the image's histogram.

    Parameters
    ----------
    img : Image
        PIL.Image object with natural  colour histogram to perform equalization (normalization)

    Returns
    -------
    Image
        Equalized (normalized) PIL.Image object
    """
    equalized = ImageOps.equalize(img)

    if np.random.rand() < 0.001:
        img.save("debug_augmentation_equalize_origin.jpg")
        equalized.save("debug_augmentation_equalize_processed.jpg")

    return equalized


def save_autonomous_image(path: str, img: Image, steering: float) -> None:
    """ This function saves an image received in autonomous mode to the specified directory for further training on it.

    Parameters
    ----------
    path : str
        Relative path to the directory where autonomous images are stored. Subdirectories are supported. For example,
        `Track-1/lap1`.
    img: Image
        PIL.Image object to save in the path directory
    steering : float
        Current steering angle associated with the image to add to `driving_log.csv` file
    """
    img_subdir: str = "IMG"
    write_mode = "a"

    if not os.path.exists(os.path.join(path, img_subdir)):
        pathlib.Path(os.path.join(path, img_subdir)).mkdir(parents=True, exist_ok=True)
        write_mode = "w"

    basename = dt.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg"
    img.save(os.path.join(path, img_subdir, basename))

    with open(os.path.join(path, "driving_log.csv"), write_mode) as fd:
        writer = csv.writer(fd)
        writer.writerow([img_subdir + "/" + basename, "", "", str(steering)])
        fd.close()


def is_autonomous_row(row: Series) -> bool:
    """ This function checks if the particular row from the dataset belongs to autonomously- or human-gathered data. In
    autonomous mode the simulator sends only center images and steering angles.

    Parameters
    ----------
    row : Series
        pandas.Series object representing a row from the dataset

    Returns
    -------
    bool
        True if the row was gathered autonomously, False otherwise
    """
    return pd.isna(row['right']) and pd.isna(row['left'])


def get_unit_of_data_from_autonomous_data(row: Series, steering: float, extra_angle: float) -> (Image, float):
    """ This function returns one image from each record in ``driving_log.csv`` file. If the row is gathered
    autonomously, it returns the only available center image. If it's a human-gathered row, it returns either left or
    right image. This function is activated by ``--train-on-autonomous-center`` argument. It can be used once there is
    data gathered with existing trained model.

    Parameters
    ----------
    row : Series
        pandas.Series object representing a row from the dataset
    steering : float
        The current steering angle associated with the image
    extra_angle : float
        The extra value needs to be added or subtracted from the current steering angle. It's used to instruct the car
        how to get back to the center of the road.

    Returns
    -------
    Image
        A picked PIL.Image object
    float
        A steering angle associated with the image with or without ``extra_angle`` value
    """
    if is_autonomous_row(row):
        image = Image.open(row['center'])
    else:
        match np.random.choice(2):
            case 0:
                image = Image.open(row['left'])
                steering += extra_angle
            case 1:
                image = Image.open(row['right'])
                steering -= extra_angle
            case _:
                raise Exception("unexpected choice")

    return image, steering


def get_unit_of_data_from_human_gathered_data(row: Series, steering: float, extra_angle: float) -> (Image, float):
    """ This function returns one image from each record in ``driving_log.csv`` file. It should be used only with a
    human-gathered dataset when all three images (center, left, and right) are presented in it.

    Parameters
    ----------
    row : Series
        pandas.Series object representing a row from the dataset
    steering : float
        The current steering angle associated with the image
    extra_angle : float
        The extra value needs to be added or subtracted from the current steering angle. It's used to instruct the car
        how to get back to the center of the road.

    Returns
    -------
    Image
        A picked PIL.Image object
    float
        A steering angle associated with the image with or without ``extra_angle`` value
    """
    match np.random.choice(3):
        case 0:
            image = Image.open(row['center'])
        case 1:
            image = Image.open(row['left'])
            steering += extra_angle
        case 2:
            image = Image.open(row['right'])
            steering -= extra_angle
        case _:
            raise Exception("unexpected choice")

    return image, steering


def get_driving_logs(dirs: list[str]) -> pd.DataFrame:
    """ This function takes a list of the relative paths to read their contents (``driving_log.csv``) and merge them
    into a single virtual dataset. It helps gather several independent datasets, merge them during training, find the
    best combination of them, and remove sub-datasets when they're not needed anymore.

    Parameters
    ----------
    dirs : list[str]
        A list of relative paths to the sub-datasets. For example: ``["Track-1/lap1", "Track-1/lap2"]``

    Returns
    -------
    DataFrame
        A pandas.DataFrame object which is a merge of one or several sub-datasets
    """
    clear_data_list: list[pd.DataFrame] = []

    for dir in dirs:
        print("Reading " + dir, file=sys.stderr)

        csv = pd.read_csv(
            os.path.join(dir, 'driving_log.csv'),
            delimiter=',',
            names=['center', 'left', 'right', 'steering'],
            usecols=[0, 1, 2, 3],
        )

        for column in ['center', 'left', 'right']:
            if csv[column].count() > 0:
                separator = "/" if "/" in csv[column][0] else "\\"
                csv[column] = csv[column].map(lambda path: os.path.join(dir, *path.split(separator)[-2:]))

        clear_data_list.append(csv)

    return pd.concat(clear_data_list, ignore_index=True)


class DataSequence(tf.keras.utils.Sequence):
    """ This class is one of the core functions. It takes a pandas.DataFrame object from ``get_driving_logs`` and
        returns preprocessed and augmented training data set along with validation dataset.

        Parameters
        ----------
        logs : DataFrame
            A product of ``get_driving_logs`` function to prepare images
        batch_size: int
            The number of images that will be used in a single forward-backward pass
        autonomous : bool
            Whether train on autonomously-gathered center images and human-gathered left and right images, or use only
            human-gathered center, left, and right images.
        extra_angle : float
            The extra value needs to be added or subtracted from a steering angle. It's used to instruct the car how to get
            back to the center of the road."""

    def __init__(self, logs: pd.DataFrame, batch_size: int, autonomous: bool, extra_angle: float, **kwargs):
        """
            """
        super().__init__(**kwargs)
        self.logs = logs
        self.batch_size = batch_size
        self.autonomous = autonomous
        self.extra_angle = extra_angle

    def __len__(self):
        return math.ceil(len(self.logs) / self.batch_size)

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        low = index * self.batch_size
        high = low + self.batch_size

        x: list[np.ndarray] = []
        y: list[np.ndarray] = []

        for _, row in self.logs.iloc[low:high].iterrows():
            steering = row['steering']

            if self.autonomous:
                image, steering = get_unit_of_data_from_autonomous_data(row, steering, self.extra_angle)
            else:
                image, steering = get_unit_of_data_from_human_gathered_data(row, steering, self.extra_angle)

            if np.random.rand() < 0.5:
                image = flip_horizontally(image)
                steering *= -1

            # if np.random.rand() < 0.5:
            #     image = blur(image)

            # if np.random.rand() < 0.5:
            #     image = grayscale(image)
            #     image = three_dimensional_grayscale(image)

            image = image.crop((
                crop_left,
                crop_top,
                origin_image_width - crop_right,
                origin_image_height - crop_bottom,
            ))

            image = equalize(image)

            # image = add_gray_layer_to_rgb_image(image)

            image = np.asarray(image)

            x.append(image)
            y.append(steering)

        return np.asarray(x), np.asarray(y)

    def on_epoch_end(self):
        self.logs = self.logs.sample(frac=1)


def build_model() -> Sequential:
    """ This function builds a CNN model which is used for steering angle predictions. It contains several convolution
    layers, dropouts to control overfitting, an input layer, several hidden layers, and an output layer.

    :return:
    """
    model = Sequential()
    model.add(Input(shape=(cropped_height(), cropped_width(), origin_colours)))

    # The first layer rescales input values from [0, 255] format to [-1, 1]
    model.add(Rescaling(1.0/127.5, offset=-1))
    # Several layers to convolve input data from (320, 80, 3) shape to (1, 15, 96) feature maps along with rectified
    # linear unit (ReLU) activation functions
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3), activation="relu"))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(3, 3), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.25))
    # Input layer
    model.add(Flatten())  # 1440 pixels
    model.add(Dropout(0.25))
    # Hidden layers with ReLU activation functions
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="relu"))
    # Output layer
    model.add(Dense(units=1))

    # Mean squared error (MeanSquaredError) loss function to calculate errors between labels and predictions
    # Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and
    # second-order moments.
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))

    return model


def draw_plot(iterations, *args):
    for i in range(0, len(args)-1, 2):
        plt.plot(range(1, iterations + 1), args[i], label=args[i+1])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best', fontsize='small')

    plt.savefig("loss-history-{}.jpg".format(started_at.strftime("%Y-%m-%d-%H-%M-%S")))


def model_callback_list() -> list[Callback]:
    list: list[Callback] = [
        ModelCheckpoint(
            "model-{}.keras".format(started_at.strftime("%Y-%m-%d-%H-%M-%S")),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        TensorBoard(
            log_dir="logs/%s" % started_at,
        ),
    ]

    return list


if __name__ == '__main__':
    started_at = dt.now()

    cli_opts = argparse.ArgumentParser()
    cli_opts.add_argument('--debug', default=True, action='store_true', help='Debug mode')
    cli_opts.add_argument('--sources', nargs='+', help='Path to datasets: --sources Track-1/f1 Track-1/b1', required=True)
    cli_opts.add_argument('--train-on-autonomous-center', default=False, action='store_true', help='Whether to use only autonomous center images or not')
    cli_opts.add_argument('--print-only', default=False, action='store_true', help='Print information on layers end exit')
    cli_opts.add_argument('--epochs', type=int, default=15, help='Number of epochs of training')
    cli_opts.add_argument('--validation-data-percent', type=float, default=0.3, help='The size of validation dataset [0, 1]')
    cli_opts.add_argument('--extra-angle', type=float, default=0.2, help='This extra value will be added when the car diverges from the center')
    options = cli_opts.parse_args()

    # Builds the CNN model for training
    model = build_model()

    if options.print_only:
        print(model.summary())
        exit(0)

    # Reads driving_log.csv files and combines them info a single pandas.DataFrame object
    logs = get_driving_logs(options.sources)

    df_val = logs.sample(frac=options.validation_data_percent)
    df_train = logs.drop(df_val.index).sample(frac=1)

    # Performs preprocessing, augmentation, and returns a list of training and validating datasets based on the combined
    # of one of several sub-datasets.
    train_sequence = DataSequence(
        df_train,
        32,
        options.train_on_autonomous_center,
        options.extra_angle,
    )

    val_sequence = DataSequence(
        df_val,
        32,
        options.train_on_autonomous_center,
        options.extra_angle,
    )

    # Trains the build model on datasets
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=options.epochs,
        batch_size=32,
        callbacks=model_callback_list(),
    )

    # Saves a history graph showing training and validation losses
    draw_plot(
        history.params['epochs'],
        history.history['val_loss'], 'Validation Loss',
        history.history['loss'], 'Training loss',
    )
