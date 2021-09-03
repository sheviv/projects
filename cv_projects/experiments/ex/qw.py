import numpy as np
import cv2
from mss import mss

sct = mss()
bounding_box = {'top': 0, 'left': 0, 'width': 800, 'height': 640}
while True:
    sct_img = sct.grab(bounding_box)
    print(f"sct_img: {type(sct_img)}")
    results = np.array(sct_img)
    print(f"sct_img: {type(sct_img)}")
    cv2.imshow('stream', results)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
