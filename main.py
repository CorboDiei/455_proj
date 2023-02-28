import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    sample_count = 0
    sample_map = {}
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("fail")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_list = []
            finger_coords = [(x + 2, x) for x in range(6, 20, 4)]
            finger_coords.extend([(x[0] + 21, x[1] + 21) for x in finger_coords])
            thumb_0 = (4, 2)
            thumb_1 = (25, 23)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_list.append((cx, cy))
            for point in hand_list:
                cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)
            count = 0
            if len(results.multi_hand_landmarks) == 2:
                mh0 = MessageToDict(results.multi_handedness[0])["classification"][0]
                l0 = mh0["index"] == 0 and mh0["label"] == "Left"
                thumb_l = thumb_0 if l0 else thumb_1
                thumb_r = thumb_1 if l0 else thumb_0
                for coord in finger_coords:
                    if hand_list[coord[0]][1] < hand_list[coord[1]][1]:
                        count += 1
                if hand_list[thumb_l[0]][0] > hand_list[thumb_l[1]][0]:
                    count += 1
                if hand_list[thumb_r[0]][0] < hand_list[thumb_r[1]][0]:
                    count += 1
                cv2.putText(image, str(count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 12)
                sample_count += 1
                sample_map[count] = sample_map.get(count, 0) + 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
        cv2.imshow("MediaPipe Hands", image)
        if sample_count == 20:
            max_val = max(sample_map, key=lambda key: sample_map[key])
            print(max_val)
            sample_count = 0
            sample_map = {}

    cap.release()

