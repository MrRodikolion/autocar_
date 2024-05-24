import ctypes
from multiprocessing import Process, Value, Array
import numpy as np
import cv2

from .road_func import (
    PID,
    Line,

)

l_th_r = 110


class RoadProcess(Process):
    def __init__(self, h, w, c, server=None):
        super().__init__()

        self.shape = (h, w, c)

        self.img_in = Array(ctypes.c_uint8, h * w * c)
        self.angle = Value('i', -1)

        self.started = Value('b', False)

        self.server = server

    def run(self):
        super().run()
        pid = PID(0.2, 0.01, 0.025)
        lin = Line(150)

        N = 470

        self.started.value = True
        while True:
            try:
                frame = np.frombuffer(self.img_in.get_obj(), dtype=np.uint8).reshape(self.shape)

                img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                _, img_gray = cv2.threshold(img_gray, l_th_r, 255, cv2.THRESH_BINARY)

                line = lin.get_line(img_gray[N], 'f')

                # print(lin.mas)

                angle = pid(line[0])

                self.angle.value = int(angle)

                # ______________________________________________________________________________________________________________
                drww = frame.copy()
                lin.draw(drww, N)

                cv2.putText(drww, str(angle), (320 + (320 - int(angle * 3.5)), N), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (200, 0, 200))
                cv2.line(drww, (320, 480), (320 + (320 - int(angle * 3.5)), N), (200, 0, 200), 1)
                # comb_img = np.hstack((drww, cv2.merge((img_gray, img_gray, img_gray))))
                if self.server:
                    np.copyto(np.frombuffer(self.server.img_road.get_obj(), dtype=np.uint8).reshape(drww.shape), drww)
                # cv2.imshow('road', comb_img)
                # cv2.waitKey(1)
                # ______________________________________________________________________________________________________________
            except BaseException as e:
                # print(e)
                pass
