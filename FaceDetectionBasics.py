import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)

            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            start_point = (int(bboxC.xmin*w), int(bboxC.ymin*h))
            end_point = (int(bboxC.xmin*w + bboxC.width*w), int(bboxC.ymin*h + bboxC.height*h))
            print('start_point', start_point)
            print('end_point', end_point)
            cv2.rectangle(img, start_point, end_point, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', start_point, cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

