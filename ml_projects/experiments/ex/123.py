import cv2
import numpy as np
from mss import mss

# sct = mss()
# bounding_box = {'top': 0, 'left': 0, 'width': 640, 'height': 480}
# bn = None
# while True:
#     sct_img = sct.grab(bounding_box)
#     print(f"sct_img: {sct_img}")
#     print(f"sct_img_size: {sct_img.size}")
#     print(f"type(sct_img): {type(sct_img)}")
#     results = np.array(sct_img)
#     if bn is None:
#         bn = results
#     print(f"results: {results}")
#     print(f"results_shape: {results.shape}")
#     print(f"type(results): {type(results)}")
#     cv2.imshow('stream', results)
#     if (cv2.waitKey(1) & 0xFF) == ord('q'):
#         cv2.destroyAllWindows()
#         break
# print(f"bn: {bn}")

visual_train_data_predict = (0, 0)
visual_train_data_predict[:, :] = np.nan
print(visual_train_data_predict)
