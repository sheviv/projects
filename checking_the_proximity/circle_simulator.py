import cv2
import numpy as np
from numpy import *

up_y = 0
ball_y = 0
new_sp = 0

plot_sp = 0
plot_pv = 0

# НАСТРОЙКИ PID CONTROLLER
h = 1.35  # 1.35
Ti = 0.15  # 0.15
Td = 1  # 1
Kp = 0.0934  # 0.0934


def pid_controller(img, y, yc, h=1, Ti=1, Td=1, Kp=1, u0=0, e0=0):
    """
    :param img:
    :param y: Measured Process Value
    :param yc: Setpoint
    :param h: Sampling Time
    :param Ti: Controller Integration Constant
    :param Td: Controller Derivation Constant
    :param Kp: Controller Gain Constant
    :param u0: Initial state of the integrator
    :param e0: Initial error
    :return:
    """
    global new_sp

    # Инициализация переменной шага
    ui_prev = u0
    e_prev = e0

    if y != yc:
        # Ошибка между желаемым и фактическим выходом
        e = yc - y
        # Интеграционный ввод   ???
        ui = ui_prev + 1.0/Ti * h*e
        # Ввод деривации   ???
        ud = 1.0/Td * (e - e_prev)/float(h)
        # Рассчитать ввод для системы   ???
        u = Kp * (e + ui + ud)
        new_sp = int(u)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h_ = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h_[0], h_[1])  # get first line
    l2 = np.cross(h_[2], h_[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z


def simulator(img, down_left_point, up_left_point, up_right_point, down_right_point):
    global up_y  # SETPOINT
    global ball_y  # FEEDBACK
    global plot_sp, plot_pv
    flag_left_dot = False
    flag_right_dot = False

    lower_center_point_1 = ((down_right_point[0] - down_left_point[0]) / 2) + down_left_point[0]
    lower_center_point_2 = (down_right_point[1] + down_left_point[1]) / 2

    upper_center_point_1 = ((up_right_point[0] - up_left_point[0]) / 2) + up_left_point[0]
    upper_center_point_2 = ((up_right_point[1] + up_left_point[1]) / 2)
    half_center_point = ((lower_center_point_2 - upper_center_point_2) / 2) + upper_center_point_2

    up_y = int(upper_center_point_1)
    up_x = int(upper_center_point_2)
    mid_x = int(half_center_point)

    # ????
    pid_controller(img, ball_y, up_y, h, Ti, Td, Kp, u0=0, e0=0)  # h=1.35, Ti=0.15, Td=1, Kp=0.0935 - Ok Settings

    # точки на боговых линиях
    support_left_dot = get_intersect(down_left_point, up_left_point, (new_sp, mid_x), (0, mid_x))
    support_right_dot = get_intersect((new_sp, mid_x), (max(up_right_point[0], down_right_point[0]) + 1, mid_x), up_right_point, down_right_point)
    if type(int(support_left_dot[0])) is int and type(int(support_left_dot[1])) is int:
        cv2.circle(img, (int(support_left_dot[0]), int(support_left_dot[1])), radius=5, color=[0, 0, 255], thickness=-1)
        flag_left_dot = True
    else:
        pass

    if type(int(support_right_dot[0])) is int and type(int(support_right_dot[1])) is int:
        cv2.circle(img, (int(support_right_dot[0]), int(support_right_dot[1])), radius=5, color=[0, 0, 255], thickness=-1)
        flag_right_dot = True
    else:
        pass

    if all([flag_left_dot, flag_right_dot]):   # ???
        # точки
        mask_support_dots = np.zeros_like(img)
        danger_area = np.array([[
            [int(down_left_point[0]), int(down_left_point[1])],
            [int(support_left_dot[0]), int(support_left_dot[1])],
            [int(support_right_dot[0]), int(support_right_dot[1])],
            [int(down_right_point[0]), int(down_right_point[1])],
        ]], dtype=np.int32)

        # маска цветовой площади дороги
        cv2.fillPoly(img, danger_area, color=(0, 0, 255))

    cv2.circle(img, (new_sp, mid_x), radius=10, color=[0, 255, 0], thickness=-1)
