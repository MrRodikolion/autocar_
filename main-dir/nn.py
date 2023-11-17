import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np
import cv2


class Net(nn.Module):
    def __init__(self, sv):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

        print('__| Loading...')
        self.load_state_dict(torch.load(sv)['state_dict'])
        self.eval()
        print('__| Loaded')

    def make_img(self, img):
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255

        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)

    def forward(self, x):
        x = self.make_img(x)

        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))

        x = x.view(-1, 16 * 12 * 12)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def net_load(filep):
    net = torch.load(filep)
    net.eval()
    return net


def Countrs(roi, l_th_nn, area=6000):  # Поиск квадратных оъектов
    Y, X, *a = roi.shape

    roi2 = cv2.resize(roi, (int(X * 2), int(Y * 2)))
    # cv2.imshow("7", roi2)
    # cv2.rectangle(frame, (380, 120), (640, 380), (0, 255, 0), 3)
    img_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    # _, img_gray = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY_INV)
    _, img_gray = cv2.threshold(img_gray, l_th_nn, 255, cv2.THRESH_BINARY)  # только темные объекты, светофоры и знаки

    img_canny = cv2.Canny(img_gray, 40, 255, 2)
    contours = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]  # 0/1
    # cv2.imshow("lk", img_canny)
    # cv2.imshow("lj", img_gray)
    # cv2.drawContours(roi2, contours, -1, (0, 0, 255 ), 1)
    # cv2.imshow("lkj", roi2)
    for i in contours:
        area_contur = cv2.contourArea(i)
        # print(area_contur)
        if (area_contur > area):  # and (area_contur < 2100):
            # print("GG",area_contur)
            # perimetr = 0.02 * cv2.arcLength(i, True)
            # approx = cv2.approxPolyDP(i, perimetr,
            #                          True)  # Если true, аппроксимируемая кривая замкнута (ее первая и последняя вершины соединены). В противном случае он не закрывается.
            # print("Количество точек в аппроксимированном контуре: " + str(len(approx)))
            rect = cv2.boundingRect(i)
            # cv2.polylines(roi2, [approx], -1, (255, 0, 0), 2)
            # cv2.rectangle(roi2, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)
            # cv2.imshow("roi2", roi2)
            # print(rect[2] / rect[3])

            if rect[2] / rect[3] < 0.7:
                sz = 0
            else:
                sz = int(area_contur // 500)  # 240

            # print(sz)
            obj = roi2[rect[1] - sz:rect[1] + rect[3] + sz, rect[0] - sz:rect[0] + rect[2] + sz]
            # search_obj = img_gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]  # одноканальное изображение
            # print(obj.shape[0] != 0)
            if obj.shape[0] != 0:
                return obj
