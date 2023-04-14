import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        # everthing which is related to the int is to put here
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection()


    def findfaces(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.faceDetection.process(imgRGB)
        #print(result)
        bboxs=[]
        if self.result.detections:
            for id,detection in enumerate(self.result.detections):
                #mpDraw.draw_detection(img,detection)
                #print(detection.location_data.relative_bounding_box)
                h, w, c = img.shape
                bboxC=detection.location_data.relative_bounding_box
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),\
                int(bboxC.width*w),int(bboxC.height*h)
                #print(bbox[0],bbox[1])
                bboxs.append([id,bbox,detection.score])


                cv2.rectangle(img, bbox, (255,0,255), 1)
                cv2.putText(img, f'{int(detection.score[0]*100)})%', (bbox[0]-30, bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
        return img,bboxs
    def fancyDraw(self,img,bbox,l=30):
        x,y,w,h=bbox
        x1,y1=x+w,y+h



def main():
    cap = cv2.VideoCapture('video.mp4')
    ptime = 0
    detector=FaceDetector()
    while True:
        success, img = cap.read()
        img,list=detector.findfaces(img)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS:{int(fps)})', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow('images', img)
        cv2.waitKey(20)


if __name__ == '__main__':
    main()
