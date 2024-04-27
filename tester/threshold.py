import cv2
import numpy as np


def on_trackbar_road(val):
    global l_th_r
    l_th_r = val


def on_trackbar_nn(val):
    global l_th_nn
    l_th_nn = val


cap = cv2.VideoCapture(0)
cv2.namedWindow('image')

cv2.createTrackbar('road', 'image', 0, 255, on_trackbar_road)
tr = cv2.createTrackbar('net', 'image', 0, 255, on_trackbar_nn)

l_th_nn, l_th_r = 160, 110
while True:
    ret, frame = cap.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    _, img_thresh_road = cv2.threshold(img_gray, l_th_r, 255, cv2.THRESH_BINARY)

    # img_thresh_net = cv2.GaussianBlur(img_gray, (3, 3), 1)
    # img_thresh_net = cv2.medianBlur(img_thresh_net, 7)
    # _, img_thresh_net = cv2.threshold(img_thresh_net, l_th_nn, 255, cv2.THRESH_BINARY)
    #
    # comb_img = np.vstack((np.hstack((cv2.merge((img_thresh_road, img_thresh_road, img_thresh_road)),
    #                                  cv2.merge((img_thresh_net, img_thresh_net, img_thresh_net)))),
    #                       np.hstack((frame, cv2.merge((img_gray, img_gray, img_gray))))))

    cv2.imshow('image', img_thresh_road)
    if cv2.waitKey(1) == ord('s'):
        print(f'l_th_nn, l_th_r = {l_th_nn}, {l_th_r}')
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
