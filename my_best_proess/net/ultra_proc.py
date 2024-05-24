import ctypes
from multiprocessing import Process, Value, Array
import numpy as np
import cv2

from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from .net_func import (
    YOLOv3,
    device,
    leanring_rate,
    load_checkpoint,
    checkpoint_file,
    ANCHORS,
    s,
    convert_cells_to_bboxes,
    nms
)

class_labels = [ 'NerovDorog', 'PrVstrech', 'SvetGo', 'SvetStop', 'park', 'stop', ]


class UltraProcess(Process):
    def __init__(self, h, w, c, server=None):
        super().__init__()

        self.shape = (h, w, c)

        self.img_in = Array(ctypes.c_uint8, h * w * c)
        self.class_id = Value('i', -1)

        self.started = Value('b', False)

        self.server = server

    def run(self):
        super().run()

        print(device)

        model = YOLO('../yolov8n_trained.pt', verbose=False)

        img = np.frombuffer(self.img_in.get_obj(), dtype=np.uint8).reshape(self.shape)
        h, w, _ = img.shape

        sh, eh = 0, h
        sw, ew = w - h, w

        while True:
            img = np.frombuffer(self.img_in.get_obj(), dtype=np.uint8).reshape(self.shape)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            img = img[sh:eh, sw:ew]
            img = cv2.resize(img, (416, 416))
            # img = img.astype(np.float32)
            # img /= 255

            net_res = model(source=img)[0]

            bboxes = net_res.boxes.data.tolist()
            if len(bboxes) > 0:
                box = max(bboxes, key=lambda x: x[4])

                conf = box[4]
                class_pred = box[5]

                self.class_id.value = int(class_pred)

                xmin, ymin, xmax, ymax = map(int, box[0:4])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (2555, 0, 0), 1)
                cv2.putText(
                    img,
                    f'{class_labels[int(class_pred)]} ({conf})',
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 1),
                    2,
                )

            if self.server:
                np.copyto(np.frombuffer(self.server.img_net.get_obj(), dtype=np.uint8).reshape(img.shape), img)
            # cv2.imshow('net', img)
            # cv2.waitKey(1)
            if not self.started.value:
                self.started.value = True
