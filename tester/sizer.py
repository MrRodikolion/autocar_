import cv2

'''
x, y = 452, 240
r = 188
'''

def f(*a):
    pass


vid = cv2.VideoCapture(1)

ret, frame = vid.read()
h, w, c = frame.shape

cv2.namedWindow('f')
cv2.createTrackbar('x', 'f', 0, w, f)
cv2.createTrackbar('y', 'f', 0, h, f)
cv2.createTrackbar('r', 'f', 0, w, f)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    x, y = cv2.getTrackbarPos('x', 'f'), cv2.getTrackbarPos('y', 'f')
    r = cv2.getTrackbarPos('r', 'f')

    cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 0, 255), 3)

    cv2.imshow('f1', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        print(f'x, y = {x}, {y}\nr = {r}')
    if key == ord('q'):
        break

