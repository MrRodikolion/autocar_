from threading import Thread
from queue import Queue

from road import *
from nn import *
from arduino import *


def capture_frame(capture, frame_queue):
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_queue[0].put(frame)
        frame_queue[1].put(frame)

    capture.release()


def road_func(frame_queue, output_queue):
    pid = PID(0.2, 0.02, 0.025)
    lin = Line(100)

    N = 470
    while True:
        frame = frame_queue.get()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, img_gray = cv2.threshold(img_gray, l_th_r, 255, cv2.THRESH_BINARY)

        line = lin.get_line(img_gray[N], 'f')

        # print(lin.mas)

        angle = pid(line[0])

        output_queue.put(angle)

        # ______________________________________________________________________________________________________________
        lin.draw(frame, N)

        cv2.putText(frame, str(angle), (320 + (320 - int(angle * 3.5)), N), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (200, 0, 200))
        cv2.line(frame, (320, 480), (320 + (320 - int(angle * 3.5)), N), (200, 0, 200), 1)
        comb_img = np.hstack((frame, cv2.merge((img_gray, img_gray, img_gray))))
        cv2.imshow('road', comb_img)
        # ______________________________________________________________________________________________________________

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_queue.task_done()
        if frame_queue.qsize() > 5:
            frame_queue.queue.clear()


def net_func(frame_queue: Queue, output_queue: Queue):
    signs = ['stop', 'pr_vstrech', 'park', 'nerov_dorog', 'svet_stop', 'svet_go', None]
    net = Net('with_svet.tar')
    indx = -1

    while True:
        frame = frame_queue.get()[100:350, 360:630]

        obj = Countrs(frame, l_th_nn)

        if obj is not None:

            try:
                # ______________________________________________________________________________________________________
                cv2.imshow('obj', obj)
                # ______________________________________________________________________________________________________
                indx = net(obj).argmax()

            except:
                pass

        output_queue.put(indx)

        # ______________________________________________________________________________________________________________
        cv2.putText(frame, signs[indx], (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
        cv2.imshow('net', frame)
        # ______________________________________________________________________________________________________________

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_queue.task_done()
        if frame_queue.qsize() > 5:
            frame_queue.queue.clear()


def comm_func(output_queues: list[Queue, Queue]):
    signs = ['stop', 'pr_vstrech', 'park', 'nerov_dorog', 'svet_stop', 'svet_go', None]
    speed = 30

    while True:
        angle, signs_indx = output_queues[0].get(), output_queues[1].get()

        print(f'{angle}\t{signs[signs_indx]}\n{output_queues[0].qsize()}\t{output_queues[1].qsize()}\n|---')

        command(s, angle, 1, speed)

        output_queues[0].task_done()
        output_queues[1].task_done()


if __name__ == '__main__':
    l_th_nn, l_th_r = 160, 110

    frame_queues = [Queue(), Queue()]
    output_queues = [Queue(), Queue()]

    video_capture = cv2.VideoCapture(0)

    capture_thread = Thread(target=capture_frame, args=(video_capture, frame_queues))
    road_thread = Thread(target=road_func, args=(frame_queues[0], output_queues[0]))
    net_thread = Thread(target=net_func, args=(frame_queues[1], output_queues[1]))
    comm_thread = Thread(target=comm_func, args=(output_queues,))

    capture_thread.start()
    road_thread.start()
    net_thread.start()
    comm_thread.start()

    capture_thread.join()
    road_thread.join()
    net_thread.join()
    comm_thread.join()

    cv2.destroyAllWindows()
