import ctypes
from multiprocessing import Process, Value, Array
import numpy as np
import cv2

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


class NetProcess(Process):
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

        load_model = True

        # Defining the model, optimizer, loss function and scaler
        model = YOLOv3(num_classes=6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=leanring_rate)
        scaler = torch.cuda.amp.GradScaler()

        # Loading the checkpoint
        if load_model:
            load_checkpoint(checkpoint_file, model, optimizer, leanring_rate)

        model.eval()

        img = np.frombuffer(self.img_in.get_obj(), dtype=np.uint8).reshape(self.shape)
        h, w, _ = img.shape

        # x, y = 452, 240
        # r = 188
        sh, eh = 0, h
        sw, ew = w - h, w

        while True:
            img = np.frombuffer(self.img_in.get_obj(), dtype=np.uint8).reshape(self.shape)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            img = img[sh:eh, sw:ew]
            img = cv2.resize(img, (416, 416))
            img = img.astype(np.float32)
            img /= 255
            try:

                x = torch.Tensor([img.transpose((2, 0, 1))]).to(device)

                with torch.no_grad():
                    # Getting the model predictions
                    output = model(x)
                    bboxes = [[] for _ in range(x.shape[0])]
                    anchors = (
                            torch.tensor(ANCHORS)
                            * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                    ).to(device)

                    # Getting bounding boxes for each scale
                    for i in range(3):
                        batch_size, A, S, _, _ = output[i].shape
                        anchor = anchors[i]
                        boxes_scale_i = convert_cells_to_bboxes(
                            output[i], anchor, s=S, is_predictions=True
                        )
                        for idx, (box) in enumerate(boxes_scale_i):
                            bboxes[idx] += box

                            # Plotting the image with bounding boxes for each image in the batch

                # bbox = max(bboxes[i], key=lambda x: x[1])
                boxes = [box for box in bboxes[0] if box[1] > 0.75]
                # boxes = nms(bboxes[0], 0.5, 0.5)
                box = max(boxes, key=lambda x: (len(tuple(filter(lambda y: y[0] == x[0], boxes))), x[1], ))
                # box = max(boxes, key=lambda x: max(filter(lambda y: y[0] == x[0], boxes))[1])
                # box = max(boxes, key=lambda x: x[1])
                # print(len([b for b in boxes if b[0] == box[0]]))

                # Getting the height and width of the image
                h, w, _ = img.shape

                # Get the class from the box
                class_pred = box[0]
                self.class_id.value = int(class_pred)

                perch = box[1]
                # Get the center x and y coordinates
                box = box[2:]
                for i in range(len(box)):
                    if box[i] > 1:
                        box[i] = 1
                # Get the upper left corner coordinates
                upper_left_x = box[0] - box[2] / 2
                upper_left_y = box[1] - box[3] / 2

                cv2.rectangle(
                    img,
                    (int(upper_left_x * w), int(upper_left_y * h)),
                    (int(upper_left_x * w + box[2] * w), int(upper_left_y * h + box[3] * h)),
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(
                    img,
                    (int(upper_left_x * w - 10), int(upper_left_y * h - 30)),
                    (int(upper_left_x * w + box[2] * w + 10), int(upper_left_y * h)),
                    (1, 1, 1),
                    -1
                )
                # print(class_labels[int(class_pred)], perch)
                cv2.putText(
                    img,
                    f'{class_labels[int(class_pred)]} ({perch})',
                    (int(upper_left_x * w), int(upper_left_y * h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 1),
                    2,
                )
            except BaseException as e:
                # print(e)
                pass

            if self.server:
                np.copyto(np.frombuffer(self.server.img_net.get_obj(), dtype=np.uint8).reshape(img.shape), cv2.cvtColor((img * 255), cv2.COLOR_BGR2RGB).astype('uint8'))
            # cv2.imshow('net', img)
            # cv2.waitKey(1)
            if not self.started.value:
                self.started.value = True
