import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture('video.mp4')
ptime=0
mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)
while True:
    success,img=cap.read()

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=faceDetection.process(imgRGB)
    #print(result)

    if result.detections:
        for id,detection in enumerate(result.detections):
            #mpDraw.draw_detection(img,detection)
            #print(detection.location_data.relative_bounding_box)
            h, w, c = img.shape
            bboxC=detection.location_data.relative_bounding_box
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),\
            int(bboxC.width*w),int(bboxC.height*h)
            print(bbox[0],bbox[1])
            cv2.rectangle(img, bbox, (255,0,255), 1)
            cv2.putText(img, f'{int(detection.score[0]*100)})%', (bbox[0]-30, bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)














    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS:{int(fps)})', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow('images', img)
    cv2.waitKey(20)