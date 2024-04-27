from road import *
from nn import *
from arduino import *


if __name__ == '__main__':
    l_th_nn, l_th_r = 160, 115
    speed = 30

    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('./out2slow.mp4')

    pid = PID(0.18, 0.015, 0.018)
    lin = Line(100)

    N = 470
    while True:
        ret, frame = video_capture.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, img_gray = cv2.threshold(img_gray, l_th_r, 255, cv2.THRESH_BINARY)

        line = lin.get_line(img_gray[N], 'f')

        # print(lin.mas)

        angle = pid(line[0])

        # ______________________________________________________________________________________________________________
        # lin.draw(frame, N)
        # try:
        #     cv2.putText(frame, str(angle), (320 + (320 - int(angle * 3.5)), N), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
        #                 (200, 0, 200))
        #     cv2.line(frame, (320, 480), (320 + (320 - int(angle * 3.5)), N), (200, 0, 200), 1)
        #     comb_img = np.hstack((frame, cv2.merge((img_gray, img_gray, img_gray))))
        #     cv2.imshow('road', comb_img)
        # except:
        #     cv2.imshow('road', frame)
        # ______________________________________________________________________________________________________________

        command(s, angle, 1, speed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
