import numpy as np
import cv2
import math
from roadmap import road_map
from circle_simulator import simulator

largestLeftLine = (0, 0, 0, 0)
largestRightLine = (0, 0, 0, 0)


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def movingAverage(avg, new_sample, N=20):
    if avg == 0:
        return new_sample
    avg -= avg / N
    avg += new_sample / N
    return avg


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    avgLeft = (0, 0, 0, 0)
    avgRight = (0, 0, 0, 0)

    # переменные состояния для отслеживания наиболее доминирующего сегмента
    largestLeftLineSize = 0
    largestRightLineSize = 0

    global largestLeftLine
    global largestRightLine

    if lines is None:
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        avgx1, avgy1, avgx2, avgy2 = avgRight
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2 - y1) / (x2 - x1))
            # фильтр по наклону и поиск наиболее доминирующего сегмента по длине
            if slope > 0.5:  # right
            # if slope > 0.5:  # right
                if size > largestRightLineSize:
                    largestRightLine = (x1, y1, x2, y2)
            elif slope < -0.5:  # left
            # elif slope < -0.5:  # left
                if size > largestLeftLineSize:
                    largestLeftLine = (x1, y1, x2, y2)

    # Определите воображаемую горизонтальную линию в центре экрана и внизу изображения,
    # чтобы экстраполировать определенный сегмент.
    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight / 3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight / 3))])
    downLinePoint1 = np.array([0, int(imgHeight)])
    downLinePoint2 = np.array([int(imgWidth), int(imgHeight)])

    # Поиск пересечения доминирующей полосы с воображаемой горизонтальной линией
    # в середине изображения и внизу изображения.
    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    up_left_point = upLeftPoint
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    down_left_point = downLeftPoint
    if math.isnan(upLeftPoint[0]) or math.isnan(downLeftPoint[0]):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        avgx1, avgy1, avgx2, avgy2 = avgRight
        return

    # рассчет и отрисовка положения средней обнаруженной левой полосы по нескольким кадрам
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (
        movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]),
        movingAverage(avgx2, downLeftPoint[0]),
        movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [31, 23, 176], 4)  # отрисовка левой линии

    # Поиск пересечения доминирующей полосы с воображаемой горизонтальной линией
    # в середине изображения и внизу изображения
    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    up_right_point = upRightPoint
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    down_right_point = downRightPoint
    if math.isnan(upRightPoint[0]) or math.isnan(downRightPoint[0]):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        avgx1, avgy1, avgx2, avgy2 = avgRight
        return

    # рассчет и отрисовка положения средней обнаруженной правой полосы по нескольким кадрам
    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]),
                movingAverage(avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [31, 23, 176], 4)  # отрисовка правой линии

    # отрисовка ограничивающих линии дороги
    road_map(img, down_left_point, up_left_point, up_right_point, down_right_point)
    # отрисовка точки направления/движения между ограничивающими линиями
    simulator(img, down_left_point, up_left_point, up_right_point, down_right_point)
