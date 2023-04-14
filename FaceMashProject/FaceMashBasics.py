''''
the idea of the module is once you have create the module you don't have to write all of these initialization and convertion
again and again so all you need to call that fun/method within our class and that will do a magic for you and will return values
for you
'''


import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
#takes RGB image only
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #process the image to find a faces
    results = faceMesh.process(imgRGB)
    #if something is detected go and draw the landmarks on the face if you have multiple face then use a for loop
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
            drawSpec,drawSpec)
            #store these landmarks only and points which will be uses for your project
        for id,lm in enumerate(faceLms.landmark):
            #print(lm)

            #takes the shape of the images
            ih, iw, ic = img.shape
            #convert the x and y values into pixels by Xing into weight and height
            x,y = int(lm.x*iw), int(lm.y*ih)
            print(id,x,y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
    3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)