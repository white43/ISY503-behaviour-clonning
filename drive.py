import base64
import glob
import os
import sys
from getopt import getopt
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

from model import crop, cropped_width, cropped_height, origin_colours, save_autonomous_image, grayscale, equalize

debug: bool = True

file: str = ""
save_image_to: str = ""
speed_limit: int = 15

grayscale_model: bool = False

cli_longopts = [
    "debug",
    "file=",
    "save-image-to=",
]

opts, args = getopt(sys.argv[1:], shortopts="", longopts=cli_longopts)

for opt, arg in opts:
    if opt == "debug":
        debug = True
    elif opt == "--file":
        file = arg
    elif opt == "--save-image-to":
        save_image_to = arg

server = socketio.Server()
app = Flask(import_name="ISY503 A3 Group 5")


def choose_throttle(speed: float) -> float:
    delta = 0.25
    throttle = 0.0

    if abs(speed_limit - speed) > delta:
        throttle = (1.0 - speed / speed_limit) * 1.5

    if 0 <= throttle < 0.09:
        throttle = 0.09
    elif -0.09 < throttle < 0:
        throttle = -0.09

    return round(throttle, 3)


def event_connect_handler(sid: str, _: dict):
    print('Connected ' + sid)


def event_telemetry_handler(sig: str, msg: dict):
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

        if np.random.rand() < 0.1:
            img.save("debug_autonomous_driving.jpg")

        image_array = np.asarray(img).reshape([1, cropped_height(), cropped_width(), origin_colours])

        predicted = model.predict(image_array, verbose=0)
        steering = round(float(predicted[0][0]), 3)

        control = {
            "steering_angle": steering.__str__(),
            "throttle": choose_throttle(speed).__str__(),
        }

        if debug is True:
            print(control)

        if save_image_to != "":
            save_autonomous_image(save_image_to, origin, steering)

    server.emit(mode, data=control, skip_sid=True)


server.on('connect', event_connect_handler)
server.on('telemetry', event_telemetry_handler)

if file == "" and os.path.exists("model.h5"):
    file = "model.h5"

if file == "":
    models = glob.glob("./model-2023-*")
    models.sort(reverse=False)
    file = os.path.basename(models[-1])

print("Model %s is chosen to be used as a target" % file)

model = load_model(file, safe_mode=False)

model_config = model.get_config()
input_shape = model_config['layers'][0]['config']['batch_input_shape']

if input_shape[3] == 1:
    origin_colours = input_shape[3]
    grayscale_model = True
    print("This model is trained on grayscale images. Colour setting have been adjusted...")


app = socketio.Middleware(server, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
