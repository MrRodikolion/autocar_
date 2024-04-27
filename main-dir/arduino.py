import serial

def constrain(val, minv, maxv):
    return min(maxv, max(minv, val))


def command(angle, dir, speed):
    comm = "SPD {},{},{} ".format(constrain(int(angle), 50, 119), dir, speed)
    # print(comm)
    if s is not None:
        s.write(comm.encode())

# s = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
# s = serial.Serial('COM1', 115200, timeout=1)
s: serial.Serial = None