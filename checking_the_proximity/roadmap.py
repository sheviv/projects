import numpy as np
import cv2


def road_map(img, down_left_point, up_left_point, up_right_point, down_right_point):
    mask = img

    # точки
    vertices = np.array([[
        [int(down_left_point[0]), int(down_left_point[1])],
        [int(up_left_point[0]), int(up_left_point[1])],
        [int(up_right_point[0]), int(up_right_point[1])],
        [int(down_right_point[0]), int(down_right_point[1])]]],
        dtype=np.int32)

    # маска цветовой площади дороги
    cv2.fillPoly(mask, vertices, 100)
    masked_image = cv2.bitwise_and(img, mask)

    for (i, j) in vertices[0]:
        cv2.circle(img, (i, j), radius=5, color=[0, 0, 255], thickness=-1)
        # circ = cv2.circle(img, (i, j), radius=10, color=[255, 0, 0], thickness=-1)
        # cv2.imwrite('circ.jpg', circ)

    return masked_image
