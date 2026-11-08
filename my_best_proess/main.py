import argparse
from pathlib import Path
from time import monotonic, sleep


def publish_frame(processes, frame, np):
    for process in processes:
        with process.img_in.get_lock():
            np.copyto(np.frombuffer(process.img_in.get_obj(), dtype=np.uint8).reshape(frame.shape), frame)


def run(camera=1, port=None, startup_timeout=30):
    import cv2
    import numpy as np
    import serial
    import torch.multiprocessing as tmp
    from net import NetProcess
    from road import RoadProcess
    from arduino import set_angle, set_state
    from net.net_func import checkpoint_file

    if not Path(checkpoint_file).is_file():
        raise RuntimeError('Missing YOLO checkpoint: ' + str(checkpoint_file))
    tmp.set_start_method('spawn', force=True)
    connection = None
    video = None
    processes = []
    try:
        if port:
            connection = serial.Serial(port, 115200, timeout=1, write_timeout=1)
        video = cv2.VideoCapture(camera)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = video.read()
        if not ok or frame is None or frame.shape != (480, 640, 3):
            raise RuntimeError('Camera must provide a valid 640x480 BGR frame')
        processes = [NetProcess(*frame.shape), RoadProcess(*frame.shape)]
        publish_frame(processes, frame, np)
        for process in processes:
            process.start()
        deadline = monotonic() + startup_timeout
        while not all(process.started.value for process in processes):
            if any(not process.is_alive() for process in processes):
                raise RuntimeError('A vision worker exited during startup')
            if monotonic() >= deadline:
                raise RuntimeError('Vision worker startup timed out')
            sleep(0.01)
        print('Vision ready; serial output ' + ('enabled' if connection else 'disabled (dry run)'))
        while True:
            ok, frame = video.read()
            if not ok or frame is None or frame.shape != (480, 640, 3):
                raise RuntimeError('Camera frame lost or dimensions changed')
            if any(not process.is_alive() for process in processes):
                raise RuntimeError('A vision worker exited')
            publish_frame(processes, frame, np)
            angle = processes[1].angle.value
            set_angle(connection, angle)
            print(angle, processes[0].class_id.value)
    finally:
        if connection is not None:
            try:
                set_state(connection, 0)
            except (serial.SerialException, OSError):
                pass
            finally:
                try:
                    connection.close()
                except (serial.SerialException, OSError):
                    pass
        if video is not None:
            video.release()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experimental 640x480 vision controller; serial disabled by default.')
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--port', help='Explicit serial device; omit for dry run')
    args = parser.parse_args(argv)
    try:
        run(args.camera, args.port)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, ImportError) as error:
        parser.exit(1, str(error) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
