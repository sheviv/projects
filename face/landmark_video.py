import sys
import cv2
import dlib
import argparse

arg_pars = argparse.ArgumentParser()
arg_pars.add_argument('-i', '--input', default='input/video_1.mp4',
                      help='path to the input video')
arg_pars.add_argument('-d', '--id', default=0,
                      help='path to the input data')
arg_pars.add_argument('-u', '--upsample', default=None, type=int,
                      help='factor by which to upsample the image, default None, ' + \
                         'pass 1, 2, 3, ...')

args = vars(arg_pars.parse_args())
args["id"] = int(args["id"])

# определить детектор лиц SVM + HOG
face_detect = dlib.get_frontal_face_detector()
# определить детектор точек лица Dlib
weight_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cap = cv2.VideoCapture(args['input'])
cap = cv2.VideoCapture(args['id'])
if not cap.isOpened():
    print('Error opening video file. Please check file path...')

# ширина и высота рамки
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


def process_boxes(box):
    """
    Вывод координат прямоуголника
    :param box:
    :return:
    """
    xmin = box.left()
    ymin = box.top()
    xmax = box.right()
    ymax = box.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


while cap.isOpened():
    # захват видеокадров
    ret, frame = cap.read()
    if ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args['upsample'] is None:
            detected_boxes = face_detect(image_rgb)
        elif 0 < args['upsample'] < 4:
            detected_boxes = face_detect(image_rgb, args['upsample'])
        else:
            # ноль - «успешное завершение программы»
            sys.exit(0)

        # перебор всех обнаруженных лиц
        for box in detected_boxes:
            shape = weight_predict(image_rgb, box)
            # отрисовка прямоуголного окна для обнаружения лиц
            res_box = process_boxes(box)
            cv2.rectangle(frame, (res_box[0], res_box[1]),
                          (res_box[2], res_box[3]), (0, 255, 0),
                          2)
            # перебор всех ключевых точек(68)
            for i in range(68):
                # отрисовка всех точек
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                           2, (0, 255, 0), -1)
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
