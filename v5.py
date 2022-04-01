import numpy as np
import argparse
import random
import time
import cv2
import os
from multiprocessing import Process

confsThr = 0.4


net = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph_coco.pb","dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

#load our input image and grab its spatial dimensions

def run_algorithm(img, numb):
        
    H, W, _ = img.shape
    # Create black image with same dimensions as input image
    black_image = np.zeros((H, W), np.uint8)
    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    detection_count = boxes.shape[2]

    for i in range(detection_count):
        box = boxes[0, 0, i]    
        classID = int(box[1])
        confidence = box[2]
        if confidence < confsThr:
            continue
        
        #Get box coordinates
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (x1, y1, x2, y2) = box.astype('int')
        roi_Width = x2 - x1
        roi_Height = y2 - y1
        
        mask = masks[i, classID]
        mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=cv2.INTER_CUBIC)
        _, mask = cv2.threshold(mask, 0.7, 255, cv2.THRESH_BINARY)
        print(black_image[y1:y2,x1:x2].shape)
        print(mask[:,:].shape)
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] != 0:
                    black_image[i+y1][j+x1] = mask[i][j]
        
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if black_image[i][j] != 255:
                img[i][j] = grey[i][j]

    cv2.imshow("Image", img)
    cv2.imshow("Black image", black_image)
    cv2.waitKey(0)
            

for i in range(2,3):
    img = cv2.imread("train_data\\"+str(i)+".jpg")
    if __name__ == '__main__':
        p1 = Process(target=run_algorithm,args=[img,i])
        p1.start()

