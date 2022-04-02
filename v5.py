from asyncio.windows_events import NULL
import numpy as np
import argparse
import random
import time
import cv2
from cv2 import imshow
from cv2 import waitKey
import os
from multiprocessing import Process

confsThr = 0.4


net = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph_coco.pb","dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

#load our input image and grab its spatial dimensions

def find_optimal_lines(ogSlice,numb):
    temp = ogSlice.copy()
    temp = cv2.GaussianBlur(ogSlice, (13,13), cv2.BORDER_CONSTANT)
    # ogSlice = cv2.Canny(ogSlice,125,150)
    temp = cv2.Canny(temp,100,175)
    size = np.size(ogSlice)
    whiteCount = np.count_nonzero(temp)
    compactness = (whiteCount/size)*100
    # print("Compactness ",compactness,"%",numb)
    if compactness < 2:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness > 4:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (9,9), cv2.BORDER_REFLECT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness > 5:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (11,11), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness < 1.15:
        temp = ogSlice.copy()
        # temp = cv2.GaussianBlur(ogSlice, (3,3), cv2.BORDER_CONSTANT)
        # threshold
        
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    print("Compactness ",compactness,"%",numb)
    return temp

# def fillSmallSpace(mask,lineImage, workerRange,x,y):
#     maskCount = 0
#     lineCount = 0
#     tempFalse = False
#     tempFalse2 = False
#     for i in range(x+1,min(x+workerRange,len(mask))):
#         if mask[i][y] != 0:
#             if tempFalse!= True:
#                 maskCount+=1 
#                 tempFalse=True
#         elif lineImage[i][y] != 0:
#             if tempFalse2 != True:
#                 lineCount+=1
#                 tempFalse2=True
                
#     tempFalse = False
#     tempFalse2 = False
#     for i in range(x-1,max(x-workerRange,-1),-1):
#         if mask[i][y] != 0:
#             if tempFalse!= True:
#                 maskCount+=1 
#                 tempFalse=True
#         elif lineImage[i][y] != 0:
#             if tempFalse2 != True:
#                 lineCount+=1
#                 tempFalse2=True
#     if maskCount == 1 and lineCount == 1:
#         return True
    
#     maskCount = 0
#     lineCount = 0
#     tempFalse = False
#     tempFalse2 = False
#     for i in range(y+1,min(y+workerRange,len(mask[0])-1)):
#         if mask[x][i] != 0:
#             if tempFalse!= True:
#                 maskCount+=1 
#                 tempFalse=True  
#         elif lineImage[x][i] != 0:
#             if tempFalse2 != True:
#                 lineCount+=1
#                 tempFalse2=True

#     tempFalse = False
#     tempFalse2 = False
#     for i in range(y-1,max(y-workerRange,-1),-1):
#         if mask[x][i] != 0:
#             if tempFalse!= True:
#                 maskCount+=1 
#                 tempFalse=True
                
#         elif lineImage[x][i] != 0:
#             if tempFalse2 != True:
#                 lineCount+=1
#                 tempFalse2=True
#     if maskCount == 1 and lineCount == 1:
#         return True
#     return False

def fillSmallSpace(mask,lineImage, workerRange,x,y):
    maskCount = 0
    lineCount = 0
    tempFalse = False
    tempFalse2 = False
    for i in range(x+1,min(x+workerRange,len(mask))):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1 
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break

    tempFalse = False
    tempFalse2 = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1 
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break
    if maskCount == 1 and lineCount == 1:
        return True
    
    maskCount = 0
    lineCount = 0
    tempFalse = False
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(mask[0])-1)):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1 
                tempFalse=True  
                break
        elif lineImage[x][i] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break

    tempFalse = False
    tempFalse2 = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1 
                tempFalse=True
                break
                
        elif lineImage[x][i] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break
    if maskCount == 1 and lineCount == 1:
        return True
    
    maskCount = 0
    lineCount = 0
    tempFalse = False
    tempFalse2 = False
    for i in range(1,min(min(workerRange+x,len(mask[0])+workerRange),min(workerRange+y,len(mask)+workerRange))):
        if mask[y+i][x+i] != 0:
            if tempFalse != True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[y+i][x+i] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break
    tempFalse = False
    tempFalse2 = False     
    for i in range(-1,max(y-workerRange,x-workerRange),-1):
        if mask[y-i][x-i] != 0:
            if tempFalse != True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[y-i][x-i] != 0:
            if tempFalse2 != True:
                lineCount+=1
                tempFalse2=True
                break
    
    return False


def run_algorithm(img, numb):
        
    H, W, _ = img.shape
    # Create black image with same dimensions as input image
    black_image = np.zeros((H, W), np.uint8)
    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    detection_count = boxes.shape[2]
    boundingBox = []
    for i in range(detection_count):
        box = boxes[0, 0, i]    
        classID = int(box[1])
        confidence = box[2]
        if confidence < confsThr:
            continue
        
        #Get box coordinates
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (x1, y1, x2, y2) = box.astype('int')
        boundingBox.append([x1,y1,x2,y2])
        roi_Width = x2 - x1
        roi_Height = y2 - y1
        
        mask = masks[i, classID]
        mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=cv2.INTER_CUBIC)
        _, mask = cv2.threshold(mask, 0.6, 255, cv2.THRESH_BINARY)
        
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] != 0:
                    black_image[i+y1][j+x1] = mask[i][j]

    maxX = 0
    maxY = 0
    minX = len(img)*len(img[0])
    minY = len(img)*len(img[0])
    for box in boundingBox:
        maxX = max(maxX,box[2])
        maxY = max(maxY,box[3])
        minX = min(minX,box[0])
        minY = min(minY,box[1])

    ogSlice = img[minY:maxY, minX:maxX]
    maskSlice = black_image[minY:maxY, minX:maxX]
    ogSlice = find_optimal_lines(ogSlice,numb)
    ogSlice = cv2.dilate(ogSlice,(3,3))
    cv2.imshow("OG Slice ", ogSlice)
    cv2.imshow("Mask Slice ", maskSlice)
    cpSlice = maskSlice.copy()#
    workerRange = int(max(len(ogSlice)/10,len(ogSlice[0])/10))
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            if maskSlice[i][j] != 255:
                workVal = fillSmallSpace(maskSlice,ogSlice,workerRange,i,j)
                if workVal == True:
                    cpSlice[i][j] = 255
                
    imshow("fillsmall",cpSlice)
    black_image_lines = black_image.copy()
    # black_image_lines.fill(0)
    for i in range(len(cpSlice)):
        for j in range(len(cpSlice[0])):
            if cpSlice[i][j] != 0:
                black_image_lines[i+minY][j+minX] = cpSlice[i][j]
    black_image = black_image_lines
    # for i in range(len(ogSlice)):
    #     for j in range(len(ogSlice[0])):
    #         if ogSlice[i][j] != 0:
    #             black_image_lines[i+minY][j+minX] = ogSlice[i][j]


    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if black_image[i][j] != 255:
                img[i][j] = grey[i][j]

    cv2.imshow("Image", img)
    # cv2.imshow("Black image", black_image)
    cv2.imshow("Black image lines", black_image_lines)
    cv2.waitKey(0)
            

for i in range(2,3):
    mask = cv2.imread("train_data\\"+str(i)+".jpg")
    if __name__ == '__main__':
        p1 = Process(target=run_algorithm,args=[mask,i])
        p1.start()

