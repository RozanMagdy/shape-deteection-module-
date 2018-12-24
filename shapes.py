import math
import numpy as np
import cv2
import imutils
tri=0
sqr=0
lin=0
circ=0 
#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 1
#camera
cap = cv2.VideoCapture(0)
print("press q to exit")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

# keep looping
while True:
    #Capture frame-by-frame
    while (True):
        ret, frame = cap.read()
        cv2.imshow("Image", frame)
        if (cv2.waitKey(1) & 0xFF == ord('m')):
            break
    if ret==True:
        #grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Canny
        canny = cv2.Canny(frame,80,240,3)
        #contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            #approximate the contour with accuracy proportional to
            #the contour perimeter
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)
            #Skip small or non-convex objects
            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue
            #triangle
            if(len(approx) == 3):
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.putText(frame, 'TRI', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
                tri=tri+1
            elif(len(approx)>=4 and len(approx)<=6):
                #nb vertices of a polygonal curve
                vtc = len(approx)
                #get cos of all corners
                cos = []
                for j in range(2,vtc+1):
                    cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                #sort ascending cos
                cos.sort()
                #get lowest and highest
                mincos = cos[0]
                maxcos = cos[-1]

                #Use the degrees obtained above and the number of vertices
                #to determine the shape of the contour
                x,y,w,h = cv2.boundingRect(contours[i])
                if(vtc==4):
                    ar = w / float(h)
		    # a square will have an aspect ratio that is approximately
		    # equal to one, otherwise, the shape is a rectangle
                    if (ar<=1.5 and ar>=0.95):
                        cv2.putText(frame,'SQR',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
                        sqr=sqr+1
                    else:
                        cv2.putText(frame,'LIN',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
                        lin=lin+1
            else:
                #detect and label circle
                area = cv2.contourArea(contours[i])
                x,y,w,h = cv2.boundingRect(contours[i])
                radius = w/2
                if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                    cv2.putText(frame,'CIRC',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
                    circ=circ+1
        #Display the resulting frame
        print("I found {} triangle ".format(tri))
        print("I found {} square ".format(sqr))
        print("I found {} line ".format(lin))
        print("I found {} circle ".format(circ))
        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)  
        #Display the resulting counts
            #Display the resulting counts triangles
        results = np.zeros((600,600,3), np.uint8)
        cv2.line(results,(75,400),(0,550),(0,0,255),6)
        cv2.line(results,(75,400),(150,550),(0,0,255),6)
        cv2.line(results,(0,550),(150,550),(0,0,255),6)
        cv2.putText(results,format(tri),(200,500),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2,cv2.LINE_AA)
            #Display the resulting lines
        cv2.line(results,(0,20),(150,20),(0,0,255),6)
        cv2.putText(results,format(lin),(200,30),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2,cv2.LINE_AA)
            #Display the resulting squares
        cv2.rectangle(results,(0,40),(150,190),(0,0,255), 6)
        cv2.putText(results,format(sqr),(200,105),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2,cv2.LINE_AA)
            #Display the resulting circles
        cv2.circle(results,(75,300), 85, (0,0,255), 6)
        cv2.putText(results,format(circ),(200,300),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('results',results)
        tri=sqr=lin=circ=0
        key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()


