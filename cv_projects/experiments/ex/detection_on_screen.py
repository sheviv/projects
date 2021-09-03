import argparse
import cv2
import exi_detect_utils
import numpy as np
import time
import torchvision
import torch

import mss
import mss.tools
# print(mss.__version__)
from mss import mss

# construct the argument parser
parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--monitor', default=1,
                    help='streaming from monitor')
# parser.add_argument('-m', '--min-size', dest='min_size', default=800,
#                     help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())
args["monitor"] = int(args["monitor"])
# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sct = mss()

def frame_mss_m(m=1):
    global sct
    if m == 1:
        monitor = {
            'top': 0,
            'left': 0,
            'width': 640,
            'height': 480
        }
    else:
        monitor_number = 2
        mon = sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"] + 10,
            "left": mon["left"] + 10,
            "width": 700,
            "height": 700,
            "mon": monitor_number,
        }
    im = np.array(sct.grab(monitor))
    im = np.flip(im[:, :, :3], 2)  # 1
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
    return True, im


frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second

while True:
    if args['monitor'] == 1:
        ret, frame = frame_mss_m(m=1)
    else:
        ret, frame = frame_mss_m(m=2)

    if ret:
        start_time = time.time()
        # load the model onto the computation device
        with torch.no_grad():
            model = model.eval().to(device)  # ?????? model.to(device).eval()
            # get predictions for the current frame
            boxes, classes, labels = exi_detect_utils.predict(frame, model, device, 0.8)
        # draw boxes and show current frame on screen
        image = exi_detect_utils.draw_boxes(boxes, classes, labels, frame)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # press `q` to exit
        wait_time = max(1, int(fps / 4))
        SIZE = 500
        # img = cv2.resize(image, (SIZE, SIZE))
        img = cv2.resize(image, (800, 800))
        cv2.imshow('Video', img)
        # cv2.imshow('Video', image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            # close all frames and video windows
            cv2.destroyAllWindows()
            break
    else:
        break
