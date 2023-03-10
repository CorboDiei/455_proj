import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import torch
import torchvision.transforms as transforms
from train_model import ConvNet
import math
from PIL import Image

def score_face(face, model):
    im = Image.fromarray(face.astype('uint8'), 'RGB')
    face = transforms.Resize((28, 28))(im)
    face = transforms.ToTensor()(face)
    print(model(face))
    return model(face)[0]


def count_fingers(hand, multi_h, image):
    mh = MessageToDict(multi_h)["classification"][0]
    left = mh["label"] == "Left"
    h, w, c = image.shape
    finger_coords = [(x + 2, x) for x in range(6, 20, 4)]
    thumb = (4, 2)
    lands = []
    count = 0
    # print(hand)
    for landmark in hand.landmark:
        cx, cy = int(landmark.x  * w), int(landmark.y * h)
        lands.append((cx, cy))
    for coord in finger_coords:
        if lands[coord[0]][1] < lands[coord[1]][1]:
            count += 1
    if left:
        if lands[thumb[0]][0] > lands[thumb[1]][0]:
            count += 1
    else:
        if lands[thumb[0]][0] < lands[thumb[1]][0]:
            count += 1
    
    return count

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    model = ConvNet()
    model.load_state_dict(torch.load('./model'))
    model.eval()
    with torch.no_grad():
        model.to(device)
        cap = cv2.VideoCapture(0)
        with mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5) as face_detection:
            with mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=5) as hands:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("fail")
                        continue

                    image = cv2.flip(image, 1)
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(image)
                    face_results = face_detection.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    count = 0

                    if face_results.detections and hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 1:

                        h, w, c = image.shape
                        best = (0, 0)
                        best_score = -20
                        hand_0, hand_1 = (0, 0), (0, 0)
                        h_0_dist, h_1_dist = 9999999, 9999999
                        
                        # find best face
                        # print(len(face_results.detections))
                        for face in face_results.detections:
                            mp_drawing.draw_detection(image, face)
                            f = MessageToDict(face)['locationData']['relativeBoundingBox']
                            
                            ul_x, ul_y = int(f['xmin'] * w), int(f['ymin'] * h)
                            lr_x, lr_y = int(f['width'] * w) + ul_x, int(f['height'] * h) + ul_y
                            if ul_x < 0 or ul_y < 0 or lr_x > w or lr_y > h:
                                continue

                            arr = image[ul_y:lr_y, ul_x:lr_x]
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                            score = score_face(arr, model)
                            print(score)
                            if score > best_score:
                                best = ((lr_x - ul_x) / 2 + ul_x, (lr_y - ul_y) / 2 + ul_y)
                                best_score = score

                        # print(best)
                        # find two closest hands
                        for idx, hand in enumerate(hand_results.multi_hand_landmarks):
                            mp_drawing.draw_landmarks(
                                image,
                                hand,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            hx, hy = int(hand.landmark[0].x * w), int(hand.landmark[0].y * h)
                            
                            dist = math.sqrt(abs(best[0] - hx) ** 2 + abs(best[1] - hy) ** 2)
                            if dist < h_0_dist:
                                hand_1 = hand_0
                                hand_0 = (hand, idx)
                                h_1_dist = h_0_dist
                                h_0_dist = dist
                            elif dist < h_1_dist:
                                hand_1 = (hand, idx)
                                h_1_dist = dist
                    
                        multi_h0, multi_h1 = hand_results.multi_handedness[hand_0[1]], hand_results.multi_handedness[hand_1[1]]
                        count = count_fingers(hand_0[0], multi_h0, image) + count_fingers(hand_1[0], multi_h1, image)
                    
                    cv2.putText(image, str(count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 12)
                    cv2.imshow("Hand Counter", image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                cap.release()

if __name__ == '__main__':
    main()
