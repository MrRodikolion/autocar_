import numpy as np
import cv2
import serial
from time import time

from net import NetProcess
from road import RoadProcess
from arduino import set_angle, set_state

current_sign = -1
timer = 0

is_back = False
is_slow = False
is_red = False

if __name__ == '__main__':
    # s = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
    # s = serial.Serial('COM5', 115200, timeout=3)
    s = None
    vid = cv2.VideoCapture(1)

    ret, frame = vid.read()
    if not ret:
        print('no cam')
        vid = cv2.VideoCapture(1)

        ret, frame = vid.read()

        if not ret:
            exit()
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    h, w, c = frame.shape

    net_p = NetProcess(h, w, c)
    road_p = RoadProcess(h, w, c)
    net_p.start()
    road_p.start()
    np.copyto(np.frombuffer(net_p.img_in.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
    np.copyto(np.frombuffer(road_p.img_in.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
    while not net_p.started.value or not road_p.started.value:
        pass
    print('started')
    while True:
        ret, frame = vid.read()
        if not ret:
            continue
        np.copyto(np.frombuffer(net_p.img_in.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)
        np.copyto(np.frombuffer(road_p.img_in.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)

        print(road_p.angle.value, net_p.class_id.value)
        set_angle(s, road_p.angle.value)

        if net_p.class_id.value != current_sign:
            current_sign = net_p.class_id.value
            if current_sign == 0:
                set_state(s, 3)
                timer = time()
                is_slow = True
            elif current_sign == 100:
                set_state(s, 5)
                timer = time()
                is_back = True
            elif current_sign == 2:
                set_state(s, 2)
                is_red = False
            elif current_sign == 3:
                set_state(s, 1)
                timer = time()
                is_red = True
            elif current_sign == 4:
                ...
            elif current_sign == 5:
                set_state(s, 0)
        else:
            if is_slow and time() - timer >= 2:
                set_state(s, 2)
                is_slow = False
            if is_back and time() - timer >= 2:
                set_state(s, 4)
                is_back = False
            if is_red and time() - timer >= 20:
                set_state(s, 2)
                is_red = False

        cv2.imshow('i', frame)
        cv2.waitKey(1)