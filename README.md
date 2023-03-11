# Corbo CSE 455 Final Project

## [Video Presentation and Demo](https://www.youtube.com/watch?v=x31C9jLM0qU)

### Problem Description

A project that I’ve been wanting to tackle for a while now is a system that lets me quickly play my favorite playlists on spotify without having to interact with a poorly executed speech recognition system, and I figured that a decent solution would be to show which playlist I want to play according to how many fingers I am holding up. In this theoretical system, a raspberry pi would be capturing video when I want to play some music, and would send those images to a CV system. This system would decipher how many fingers I’m holding up, send that info to a playlist mapping server which would then ping the Alexa API to play that playlist on the speakers in my room.

### Approach (Pipeline)

This project only concerns the computer vision portion of this system. This is a walkthrough of the pipeline:

1. Mediapipe finds all the bounding boxes for faces in a particular frame

2. Those faces are all fed through a convolutional neural network to decide which is the closest to my face

3. Mediapipe finds the positions of all the hands in the image

4. The two closest hands to my face (by L2 norm) are chosen to be mine

5. The relative positions of finger tips and knuckles of each finger on each hand is used to determine whether it is raised or not

6. The finger count is added up and displayed

### Previous Work

I utilized the following pre-written code to fascilitate my job in this project:

- pytorch (torchvision) 

- Mediapipe (the face and hand positional detection libraries)

- An approximate LeNet5 architecture that is close to the one in [this paper](https://www.tandfonline.com/doi/full/10.1080/21642583.2020.1836526#:~:text=CNN%20model%20for%20face%20recognition,width%20and%20full%20connection%20layer.)

### Dataset

I partially created and curated the dataset I used to train the CNN. All of the images of myself come from my camera and were processed through the Mediapipe face detection pipeline. The not_corbo faces come from [this kaggle dataset](https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download) and again were processed through the Mediapipe face detection pipeline.

corbo n=129
not_corbo n=5640

### Results and Discussion

Originally I was just going to do teh finger counting part of the pipeline, but I found that when I had friends over the system would start choosing their hands even if they were farther away from the camera than me, so I had to increase the complexity of the system by quite a bit with the facial recognition system. Doing this, however, also let me explore more parts of the machine learning aspect of the class.

The system for the most part chooses my face and thus my hands. Sometimes mediapipe either misses a hand or doubles up on the hand detection for a few frames before sorting out. To help smooth out the final output then, I could do some batching and choose the mode finger count.

I will probably continue to integrate this project into the system I mention at the beginning. To continue this I will need to build a raspberry pi system with a camera and a decent graphics system to perform the operations contained in thsi project. Then I will need to create a system that takes that output and pings the Alexa API based on which playlist is being indicated. 