import numpy as np
import cv2

from time import time

abs_centr = 290

class Line:
    def __init__(self, z):
        '''
        :param z: halp of line
        '''
        self.norm = abs_centr
        self.z = z

        self.lin = None
        self.centr_old = np.array([320, 20])

        self.r = 0
        self.mas = None

    def find_centers(self, img):
        self.mas = np.zeros((640, 2), dtype='int16')

        self.r = 0
        c = self.norm - self.z
        while c < self.norm + self.z:
            e = c

            # cv2.circle(img_copy, (c, 400), 2, (0, 255, 0))

            while e < 640 and img[e] < 255:
                e += 1

            lin = e - c
            if lin > 0:
                centr = int((c + e) / 2)
                # cv2.circle(img_copy, (centr, 395), 2, (0, 100, 255))

                self.mas[self.r][0], self.mas[self.r][1] = centr, lin
                self.r += 1
                # print(mas)
            c = e + 1

    def get_line(self, img, go):
        self.find_centers(img)

        if go == 'f':
            self.lin = min(self.mas, key=lambda x: abs(x[0] - self.centr_old[0]) if x[0] > 0 else 640)
        elif go == 'l':
            self.lin = self.mas[0]
        elif go == 'r':
            self.lin = self.mas[self.r - 1]
        # print(self.lin)
        if self.lin[1] > 300 or self.lin[0] == 0:
            '''print(self.lin, self.centr_old)
            cv2.putText(img_copy, str(self.lin), (self.lin[0], 420), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (200, 0, 200))
            cv2.circle(img_copy, (self.lin[0], 410), 5, (0, 200, 200))'''

            # if 100 < self.lin[1] < 190 and self.lin[0] > 350:
            #     print('stop lin')
            # elif self.lin[1] >= 190:
            #     print('perf')

            self.lin = self.centr_old

        self.norm = int(abs_centr - ((abs_centr - self.lin[0]) // 1.5))
        self.centr_old = self.lin
        return self.lin

    def draw(self, img, N):
        cv2.line(img, (self.lin[0], N), (self.lin[0], N - 10), (0, 0, 255), 15)
        cv2.line(img, (self.norm - self.z, N), (self.norm + self.z, N), (0, 0, 255), 5)
        cv2.line(img, (abs_centr, N), (abs_centr, N + 10), (255, 0, 0), 3)


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd

        self.prev_error = 0
        self.integral = 0

        self.st_t = time()

    def __call__(self, centr):
        error = abs_centr - centr
        self.integral += error * self.ki # * (self.st_t - time())
        self.integral = self.integral / (self.st_t - time())
        # integral = constrain(integral, 50, 500)
        # self.integral = max(-20, min(20, self.integral))

        derivative = error - self.prev_error

        angle = int(self.kp * error + self.integral + self.kd * derivative) + 90

        self.prev_error = error

        # print(f'<------->\n_> angle: {angle}\n_> err: {error} | integ: {self.integral} | deriv: {derivative}')
        # self.st_t = time()
        return angle


if __name__ == '__main__':
    pass
