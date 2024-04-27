import serial


# s = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
# s = serial.Serial('COM5', 115200, timeout=2)


def set_angle(s, angle):
    '''
    :param angle: from 50 to 129
    :return:
    '''
    if s is None:
        return -1
    str_angle = '$A{},'.format(angle)
    s.write(str_angle.encode())
    s.read()


def set_state(s, state):
    '''
    :param state: from 0 to 9
    '''
    if s is None:
        return -1
    str_state = '$T{},'.format(state)
    s.write(str_state.encode())
    s.read()
