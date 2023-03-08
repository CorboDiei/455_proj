import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from PIL import Image
from os import listdir
from os.path import isfile, join

mp_face_detection = mp.solutions.face_detection
map_drawing = mp.solutions.drawing_utils
# p = './dataset/original/'
# image_files = [f for f in listdir(p) if isfile(join(p, f))]

with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
) as face_detection:
    cap = cv2.VideoCapture(0)
    count = 196
    # for idx, file in enumerate(image_files):
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        count += 1
        # if not file.endswith(".jpg"):
        #     continue
        # print(file)
        # image = cv2.imread(join(p, file))
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            res = MessageToDict(results.detections[0])['locationData']['relativeBoundingBox']
            h, w, c = image.shape
            ul_x, ul_y = int(res['xmin'] * w), int(res['ymin'] * h)
            lr_x, lr_y = int(res['width'] * w) + ul_x, int(res['height'] * h) + ul_y
            if ul_x < 0 or ul_y < 0 or lr_x > w or lr_y > h:
                continue
            # print("{} {} {} {}".format(ul_x, ul_y, lr_x, lr_y))
            im_arr = image[ul_y:lr_y, ul_x:lr_x]
            im = Image.fromarray(cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB))
            if count < 404 and count % 3 == 0:
                im.save("dataset/corbo/im{}.jpg".format(count))
            for detection in results.detections:
                map_drawing.draw_detection(image, detection)

        cv2.imshow("MediaPipe Face Detection", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()