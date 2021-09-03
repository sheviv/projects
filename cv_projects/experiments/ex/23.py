"""
argparse
"""
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--vbn", help="increase output verbosity")
# args = parser.parse_args()
# if args.vbn:
# print(f"vbn vbn vbn: {args.vbn}")
# or
# args = vars(parser.parse_args())
# if args["vbn"]:
#     print(f"vbn vbn vbn: {args['vbn']}")

"""
streaming(mss library) screen
"""
import cv2
import numpy as np
from mss import mss
sct = mss()
bounding_box = {'top': 0, 'left': 0, 'width': 400, 'height': 300}
while True:
    sct_img = sct.grab(bounding_box)
    results = np.array(sct_img)
    cv2.imshow('stream', results)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
