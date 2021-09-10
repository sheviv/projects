import numpy as np
import cv2
import mss
# print(mss.__version__)
from mss import mss
from PIL import Image
# mon = {'top': 0, 'left': 0, 'width': 480, 'height': 640}
# sct = mss()
# while 1:
#  sct.get_pixels(mon)
#  # sct.getpixel(mon)
#  img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
#  cv2.imshow('test', np.array(img))
#  if cv2.waitKey(25) & 0xFF == ord('q'):
#   cv2.destroyAllWindows()
#   break



import mss
import mss.tools
# with mss.mss() as sct:
#     # Get information of monitor 2
#     monitor_number = 2
#     mon = sct.monitors[monitor_number]
#     # The screen part to capture
#     monitor = {
#         "top": mon["top"] + 100,  # 100px from the top
#         "left": mon["left"] + 100,  # 100px from the left
#         "width": 160,
#         "height": 135,
#         "mon": monitor_number,
#     }
#     # output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)
#     # Grab the data
#     sct_img = sct.grab(monitor)

# monitor = {
#         "top": mon["top"] + 100,  # 100px from the top
#         "left": mon["left"] + 100,  # 100px from the left
#         "width": 160,
#         "height": 135,
#         "mon": monitor_number,
#     }

# sct = mss()
while True:
    with mss.mss() as sct:
     monitor_number = 2
     mon = sct.monitors[monitor_number]
     monitor = {
      "top": mon["top"] + 10,  # 100px from the top
      "left": mon["left"] + 10,  # 100px from the left
      "width": 500,
      "height": 500,
      "mon": monitor_number,
     }
    sct_img = sct.grab(monitor)
    # print(f"sct_img: {type(sct_img)}")
    results = np.array(sct_img)
    # print(f"sct_img: {type(sct_img)}")
    cv2.imshow('stream', results)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
