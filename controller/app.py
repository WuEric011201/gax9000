import os
import json
import logging
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS  # Disable on deployment
from gevent.pywsgi import WSGIServer
import gevent
import sse
from controller.util import timestamp_date
from controller.backend import Controller, ControllerSettings, ControllerApiHandler, UserProfile, MonitorApiHandler

def save_default_settings(path_settings):
    controller_settings = ControllerSettings.default()
    with open(os.path.join(path_settings, "config.json"), "w+") as f:
        json.dump(controller_settings.__dict__, f, indent=2)

    path_users = os.path.join(path_settings, "users")
    os.makedirs(path_users, exist_ok=True)

    for username in controller_settings.users:
        UserProfile.default(username).save(path_users)

def create_server(path_settings, cors=True):
    path_controller_settings = os.path.join(path_settings, "config.json")
    if not os.path.exists(path_controller_settings):
        logging.info(f"Generating new default config in settings path: \"{path_settings}\"")
        save_default_settings(path_settings)

    path_users = os.path.join(path_settings, "users")

    channel_controller = sse.EventChannel()
    channel_monitor = sse.EventChannel()

    controller = Controller(
        path_settings=path_controller_settings,
        path_users=path_users,
        monitor_channel=channel_monitor,
    )

    app = Flask(__name__)
    if cors:
        CORS(app)

    @app.route("/subscribe")
    def subscribe():
        return channel_monitor.subscribe()

    @app.route("/publish", methods=["POST"])
    def publish():
        channel_monitor.publish(request.data)
        return "OK"

    @app.route("/")
    def index():
        return """<body><script>
    var eventSource = new EventSource('/subscribe');
    eventSource.onmessage = function(m) {
        console.log(m);
        var el = document.getElementById('messages');
        el.innerHTML += m.data;
        el.innerHTML += '</br>';
    }
    function post(url, data) {
        var request = new XMLHttpRequest();
        request.open('POST', url, true);
        request.setRequestHeader('Content-Type', 'text/plain; charset=UTF-8');
        request.send(data);
    }
    function publish() {
        var message = document.getElementById('msg').value;
        post('/publish', message);
    }
    </script>
    <input type='text' id='msg'>
    <button onclick='publish()'>send</button>
    <p id='messages'></p>
    </body>"""

    @app.route("/event/controller")
    def event_controller():
        return channel_controller.subscribe()

    @app.route("/event/monitor")
    def event_monitor():
        return channel_monitor.subscribe()

    api = Api(app)
    api.add_resource(ControllerApiHandler, "/api/controller", resource_class_kwargs={
        "channel": channel_controller,
        "monitor_channel": channel_monitor,
        "controller": controller,
    })
    api.add_resource(MonitorApiHandler, "/api/monitor", resource_class_kwargs={
        "channel": channel_monitor,
    })

    return app

def run(port=9000, path_settings="./settings"):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)
    logFileHandler = logging.FileHandler(f"logs/{timestamp_date()}.log")
    logFileHandler.setLevel(logging.DEBUG)
    logFileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(logFileHandler)

    logConsoleHandler = logging.StreamHandler()
    logConsoleHandler.setLevel(logging.DEBUG)
    logConsoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(logConsoleHandler)

    logging.info("============================================================")
    logging.info("RUNNING GAX 9000")
    logging.info("============================================================")
    logging.info(f"Settings path: \"{path_settings}\"")

    app = create_server(path_settings=path_settings)

    path_cert = os.path.join(path_settings, "ssl", "cert.pem")
    path_key = os.path.join(path_settings, "ssl", "key.pem")

    if os.path.exists(path_cert) and os.path.exists(path_key):
        logging.info("Starting with SSL")
        server = WSGIServer(("", port), app, certfile=path_cert, keyfile=path_key)
    else:
        logging.warning("SSL cert/key not found. Starting in HTTP mode.")
        server = WSGIServer(("", port), app)

    logging.info(f"Controller server listening on port: {port}")
    server.serve_forever()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run gax controller server.")
    parser.add_argument("path_settings", type=str, help="Controller config data path")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind the server to")
    args = vars(parser.parse_args())
    run(**args)
