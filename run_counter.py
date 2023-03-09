import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_model import ConvNet

def score_face(face, model):
    print('ye2')

def count_fingers(hand):
    print('ye3')

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvNet()
    model.load_state_dict(torch.load('./model'))
    model.eval()
    model.to(device)
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5) as face_detection:
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
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

                for face in face_results.detections:
                    

    

if __name__ == '__main__':
    main()
