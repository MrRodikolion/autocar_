import ctypes
from multiprocessing import Process, Value, Array
import numpy as np
import cv2
from flask import Flask, Response, render_template, url_for, send_file
import asyncio
import io
import logging


class ServerProcess(Process):
    def __init__(self, h, w, c):
        super().__init__()

        self.shape = (h, w, c)
        self.img_norm = Array(ctypes.c_uint8, h * w * c)
        self.img_road = Array(ctypes.c_uint8, h * w * c)
        self.img_net = Array(ctypes.c_uint8, 416 * 416 * 3)

    def run(self):
        super().run()

        app = Flask(__name__, template_folder=r"templaytes")
        app.debug = False
        app.logger.disabled = True
        log = logging.getLogger('werkzeug')
        log.disabled = True

        def gen():
            while True:
                frame = np.frombuffer(self.img_norm.get_obj(), dtype=np.uint8).reshape(self.shape)
                _, bufframe = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bufframe.tobytes() + b'\r\n\r\n')

        @app.route('/video_feed')
        def video_feed():
            return Response(gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/norm')
        def norm():
            frame = np.frombuffer(self.img_norm.get_obj(), dtype=np.uint8).reshape(self.shape)
            _, bufframe = cv2.imencode('.jpg', frame)
            img_io = io.BytesIO(bufframe.tobytes())

            return send_file(img_io, mimetype='image/jpeg')

        @app.route('/road')
        def road():
            frame = np.frombuffer(self.img_road.get_obj(), dtype=np.uint8).reshape(self.shape)
            _, bufframe = cv2.imencode('.jpg', frame)
            img_io = io.BytesIO(bufframe.tobytes())

            return send_file(img_io, mimetype='image/jpeg')

        @app.route('/net')
        def net():
            frame = np.frombuffer(self.img_net.get_obj(), dtype=np.uint8).reshape((416, 416, 3))
            _, bufframe = cv2.imencode('.jpg', frame)
            img_io = io.BytesIO(bufframe.tobytes())

            return send_file(img_io, mimetype='image/jpeg')

        @app.route('/')
        def index():
            return render_template('index.html')

        app.run(host='192.168.3.19', port=5000, ssl_context="adhoc")
