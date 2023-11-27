import eventlet.wsgi
import socketio
from flask import Flask

debug = True
file = None

server = socketio.Server()
app = Flask(import_name="ISY503 A3 Group 5")


def event_connect_handler(sid: str, _: dict):
    print('Connected ' + sid)


def event_telemetry_handler(sig: str, msg: dict):
    mode = "manual"
    control = {}

    if msg:
        mode = "steer"

        control = {
            "steering_angle": "0",
            "throttle": "1",
        }

        if debug is True:
            print(control)

    server.emit(mode, data=control, skip_sid=True)


server.on('connect', event_connect_handler)
server.on('telemetry', event_telemetry_handler)

if __name__ == '__main__':
    app = socketio.Middleware(server, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
