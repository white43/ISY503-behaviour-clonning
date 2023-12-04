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
from keras.models import load_model

from model import crop, cropped_width, cropped_height, origin_colours, save_autonomous_image, grayscale, equalize

sum_error: float = 0.0

grayscale_model: bool = False

server = socketio.Server()
app = Flask(import_name="ISY503 A3 Group 5")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument('--debug', default=True, action='store_true', help='Debug mode')
cli_opts.add_argument('--file', type=str, default='', help='Path to file with saved model')
cli_opts.add_argument('--save-image-to', type=str, default='', help='Path to directory where to save autonomous data')
cli_opts.add_argument('--speed_limit', type=float, default=10.0, help='The model will try to not exceed this speed')
options = cli_opts.parse_args()


def choose_throttle(speed: float, speed_limit: float) -> float:
    global sum_error
    delta = 0.25
    throttle = 0.0

    sum_error += speed_limit - speed

    if abs(speed_limit - speed) > delta:
        throttle = ((1.0 - speed / speed_limit) * 1.5) + (sum_error * 0.0025)

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
    global sum_error, options

    mode: str = "manual"
    control: dict[str, str] = {}

    if msg:
        mode = "steer"

        img_center = msg["image"]
        speed = float(msg["speed"])
        img = Image.open(BytesIO(base64.b64decode(img_center)))
        origin = img

        img = crop(img)
        img = equalize(img)

        if grayscale_model:
            img = grayscale(img)

        if options.debug is True and np.random.rand() < 0.1:
            img.save("debug_autonomous_driving.jpg")

        image_array = np.asarray(img).reshape([1, cropped_height(), cropped_width(), origin_colours])

        predicted = model.predict(image_array, verbose=0)
        steering = round(float(predicted[0][0]), 3)

        control = {
            "steering_angle": steering.__str__(),
            "throttle": choose_throttle(speed, options.speed_limit).__str__(),
            "sum_error": sum_error,
        }

        if options.debug is True:
            print(control)

        if options.save_image_to != "":
            save_autonomous_image(options.save_image_to, origin, steering)

    server.emit(mode, data=control, skip_sid=True)


server.on('connect', event_connect_handler)
server.on('telemetry', event_telemetry_handler)

if options.file == "" and os.path.exists("model.h5"):
    options.file = "model.h5"

if options.file == "":
    models = glob.glob("./model-2023-*")
    models.sort(reverse=False)
    options.file = os.path.basename(models[-1])

print("Model %s is chosen to be used as a target" % options.file)

model = load_model(options.file, safe_mode=False)

model_config = model.get_config()
input_shape = model_config['layers'][0]['config']['batch_input_shape']

if input_shape[3] == 1:
    origin_colours = input_shape[3]
    grayscale_model = True
    print("This model is trained on grayscale images. Colour setting have been adjusted...")


app = socketio.Middleware(server, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
