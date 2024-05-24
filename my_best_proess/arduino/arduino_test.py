import time

import serial

s_base = '$A{},'
s_type = '$T{},'

# s = serial.Serial('COM5', 115200, timeout=2)
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
time.sleep(3)

# s.write(s_base.format(135).encode())
# s.read()
# time.sleep(1)

# print('stop')
# s.write(s_type.format(0).encode())
# s.read()
# time.sleep(1)
#
# print('nerov_d')
# s.write(s_type.format(2).encode())
# s.read()
# time.sleep(2)
# s.write(s_type.format(1).encode())
# s.read()
# time.sleep(1)
#
# print('forward/backward')
# s.write(s_type.format(4).encode())
# s.read()
# time.sleep(3)
# s.write(s_type.format(3).encode())
# s.read()
# time.sleep(1)
#
# while True:
#     s.write(s_type.format(0).encode())
#     s.read()
#     time.sleep(2)
#
#     s.write(s_type.format(1).encode())
#     s.read()
#     time.sleep(2)

# while True:
#     s.write(s_base.format(135).encode())
#     s.read()
#     s.write(s_type.format(0).encode())
#     s.read()
#     time.sleep(1)
#
#     s.write(s_base.format(45).encode())
#     s.read()
#     s.write(s_type.format(1).encode())
#     s.read()
#     time.sleep(1)

while True:
    s.write(s_type.format(0).encode())
    s.read()
    for a in range(60, 120):
        print(a)
        s.write(s_base.format(a).encode())
        s.read()
        time.sleep(0.01)

    s.write(s_type.format(1).encode())
    s.read()
    for a in range(120, 60, -1):
        print(a)
        s.write(s_base.format(a).encode())
        s.read()
        time.sleep(0.01)