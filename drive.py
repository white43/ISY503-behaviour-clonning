import argparse
import base64
import glob
import os
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.api.models import load_model

from model import crop, cropped_width, cropped_height, origin_colours, save_autonomous_image, grayscale, equalize

grayscale_model: bool = False

# Initialize Socker.IO web server and Flask web application
server = socketio.Server()
app = Flask(import_name="ISY503 A3 Group 5")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument('--debug', default=True, action='store_true', help='Debug mode')
cli_opts.add_argument('--file', type=str, default='', help='Path to file with saved model')
cli_opts.add_argument('--save-image-to', type=str, default='', help='Path to directory where to save autonomous data')
cli_opts.add_argument('--speed-limit', type=float, default=12.0, help='The model will try to not exceed this speed')
options = cli_opts.parse_args()


def choose_throttle(speed: float, speed_limit: float) -> float:
    delta = 0.25
    throttle = 0.0

    if abs(speed_limit - speed) > delta:
        throttle = ((1.0 - speed / speed_limit) * 1.5)

        if throttle > 1:
            throttle = 1

    if 0 <= throttle < 0.09:
        throttle = 0.09
    elif -0.09 < throttle < 0:
        throttle = -0.09

    return round(throttle, 3)


def event_connect_handler(sid: str, _: dict):
    print('Connected ' + sid)


def event_telemetry_handler(sig: str, msg: dict):
    global options

    mode: str = "manual"
    control: dict[str, str] = {}

    if msg:
        mode = "steer"

        # Read input data from JSON
        img_center = msg["image"]
        speed = float(msg["speed"])
        # Convert base64-encoded image to PIL.Image object
        img = Image.open(BytesIO(base64.b64decode(img_center)))
        origin = img

        # Preprocessing
        img = crop(img)
        img = equalize(img)

        # Convert image to grayscale palette if the model was trained on grayscale images
        if grayscale_model:
            img = grayscale(img)

        if options.debug is True and np.random.rand() < 0.1:
            img.save("debug_autonomous_driving.jpg")

        # Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 80, 320, 3), found
        # shape=(None, 320, 3)
        image_array = np.asarray(img).reshape([1, cropped_height(), cropped_width(), origin_colours])

        # Getting steering angles from the trained model
        predicted = model.predict(image_array, verbose=0)
        steering = round(float(predicted[0][0]), 3)

        control = {
            "steering_angle": steering.__str__(),
            "throttle": choose_throttle(speed, options.speed_limit).__str__(),
        }

        if options.debug is True:
            print(control)

        # Save autonomous images for further training on them
        if options.save_image_to != "":
            save_autonomous_image(options.save_image_to, origin, steering)

    # Send answer back to the simulator
    server.emit(mode, data=control, skip_sid=True)


# Associate Socket.IO's frames with their handlers
server.on('connect', event_connect_handler)
server.on('telemetry', event_telemetry_handler)

if options.file == "" and os.path.exists("model.keras"):
    options.file = "model.keras"

# Choose the last saved model on disk
if options.file == "":
    models = glob.glob("./model-202?-*.keras")
    models.sort(reverse=False)

    if len(models) > 0:
        options.file = os.path.basename(models[-1])

if options.file == "":
    print("No trained model found to drive")
    exit(1)

print("Model %s is chosen to be used as a target" % options.file)

# Loading saved model. There will be errors without safe_mode=False.
model = load_model(options.file, safe_mode=False)

model_config = model.get_config()
input_shape = model_config['layers'][0]['config']['batch_shape']

# If we're using a grayscale-model, we will need to convert input images to grayscale as well
if input_shape[3] == 1:
    origin_colours = input_shape[3]
    grayscale_model = True
    print("This model is trained on grayscale images. Colour setting have been adjusted...")


# Listen to 4567 port which is used to receive input data from the simulator
app = socketio.Middleware(server, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
