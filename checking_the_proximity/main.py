import numpy as np
import cv2
import mss
from image_processing import process_image
import time
import warnings
warnings.filterwarnings('ignore')

sct = mss.mss()


def img_proccessor(state):
    while state:

        last_time = time.time()
        monitor = {'top': 65, 'left': 65, 'width': 1350, 'height': 960}
        original_img = np.array(sct.grab(monitor))
        resized_img = cv2.resize(original_img, (1200, 800))

        fps_calc = 1 / (time.time() - last_time)
        fps = ('fps: {0}'.format(int(fps_calc)))

        new_img = cv2.putText(process_image(resized_img),
                              fps,
                              (20, 20),
                              fontFace=4,
                              color=[0, 255, 0],
                              thickness=1,
                              fontScale=0.8)  # Color = BGR
        cv2.imshow('Processed', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
img_proccessor(True)
