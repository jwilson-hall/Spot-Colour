#C1949699
#
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
boxThr = 0.45
#Loading in rcnn mask model
net = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph_coco.pb","dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
#Loading in bounding box model
net2 = cv2.dnn_DetectionModel("frozen_inference_graph.pb","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
net2.setInputSize(320,320)
net2.setInputScale(1.0/127.5)
net2.setInputMean((127.5,127.5,127.5))
net2.setInputSwapRB(True)
#load our input image and grab its spatial dimensions

def find_optimal_lines(ogSlice,numb):#attempts to find the best parameters to use for the canny edge detector
    temp = ogSlice.copy()
    temp = cv2.GaussianBlur(ogSlice, (13,13), cv2.BORDER_CONSTANT)
    # ogSlice = cv2.Canny(ogSlice,125,150)
    temp = cv2.Canny(temp,100,175)
    size = np.size(temp)
    whiteCount = np.count_nonzero(temp)
    compactness = (whiteCount/size)*100
    # while (compactness < 6 or compactness > 9):
    if compactness < 3:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
    if compactness < 3.5:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        # threshold        
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
    if compactness > 8:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (7,7), cv2.BORDER_REFLECT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
    if compactness > 9:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (9,9), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
    if compactness < 6:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        # threshold        
        temp = cv2.Canny(temp,150,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
    return temp

def fillSmallSpace(img, workerRange,x,y):#function fills in small gaps between 2 white pixels
    count = 0
    down = 0
    up = 0
    left = 0
    right = 0
    tempFalse = False
    for i in range(x+1,min(x+workerRange,len(img))):
        if img[i][y] != 0:
            if tempFalse!= True:
                count+=1 
                down+=1
                tempFalse=True
                break    
    tempFalse = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if img[i][y] != 0:
            if tempFalse != True:
                count+=1
                up+=1
                tempFalse=True
                break
    if count == 2:
        return count
    else:
        count = 0
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(img[0])-1)):
        if img[x][i] != 0:
            if tempFalse!= True:
                count+=1
                right+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if img[x][i] != 0:
            if tempFalse != True:
                count+=1
                left+=1
                tempFalse=True
                break
    return count

def smoothing(mask,lineImage, workerRange,x,y):#function looking for neighbours in 4 directions
    maskCount = 0
    lineCount = 0
    tempFalse = False
    for i in range(x+1,min(x+workerRange,len(mask))):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break    
    
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(mask[0])-1)):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True  
                break
        elif lineImage[x][i] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break

    tempFalse = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break      
        elif lineImage[x][i] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break

    return lineCount,maskCount

def run_algorithm(img, numb):
        
    H, W, _ = img.shape
    # Create black image with same dimensions as input image
    black_image = np.zeros((H, W), np.uint8)# creates black image from input image so it has the same dimensions
    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])#parses the image top the network and gets the binary mask
    _, _, boxes2 = net2.detect(img,confThreshold = boxThr)#parses the image top the network and gets the output boxes

    detection_count = boxes.shape[2]
    boundingBox = []
    for i in range(detection_count):#looks through all the detections of the network and skips over the ones that are below the threshold
        box = boxes[0, 0, i]    
        classID = int(box[1])
        confidence = box[2]
        if confidence < confsThr:
            continue
        #Get box coordinates
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (x1, y1, x2, y2) = box.astype("int")
        # boundingBox.append([x1,y1,x2,y2])
        roi_Width = x2 - x1
        roi_Height = y2 - y1
        
        mask = masks[i, classID]
        mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=cv2.INTER_CUBIC)
        _, mask = cv2.threshold(mask, 0.6, 255, cv2.THRESH_BINARY)
        
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] != 0:
                    black_image[i+y1][j+x1] = mask[i][j]

    for box in boxes2:
        boundingBox.append([box[0],box[1],box[2]+box[0],box[3]+box[1]])
    maxX = 0
    maxY = 0
    minX = len(img)*len(img[0])
    minY = len(img)*len(img[0])
    for box in boundingBox:#finds the bounding region of all the bounding boxes 
        maxX = max(maxX,box[2])
        maxY = max(maxY,box[3])
        minX = min(minX,box[0])
        minY = min(minY,box[1])

    ogSlice = img[minY:maxY, minX:maxX]#slice is set to the bounding area
    maskSlice = black_image[minY:maxY, minX:maxX]
    ogSlice = (ogSlice,numb)
    ogSlice = cv2.dilate(ogSlice,(5,5),iterations=3)
    # cv2.imshow("OG Slice ", ogSlice)
    newSlice = ogSlice.copy()
    newSlice.fill(0)

    for b in boundingBox:
        for i in range(b[0]-minX,(b[2])-minX):
            for j in range(b[1]-minY,(b[3])-minY):
                newSlice[j][i] = ogSlice[j][i]
    
    ogSlice = newSlice
    # cv2.imshow("OG Slice ", ogSlice)
    cpSlice = maskSlice.copy()#
    
    workerRange = int(max(len(ogSlice)/10,len(ogSlice[0])/10))
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            if maskSlice[i][j] != 255:
                workVal = smoothing(maskSlice,ogSlice,workerRange,i,j)
                if (workVal[0] >= 2 and workVal[1] >= 2) or (workVal[0] > 3):
                    cpSlice[i][j] = 255
                # elif workVal[0] > 3:
                #     cpSlice[i][j] = 255
    # ogSlice = cpSlice.copy()
    cpSlice2 = cpSlice.copy()
    cpSlice = cv2.bitwise_or(maskSlice,ogSlice)#combining the images
    cpSlice = cv2.bitwise_or(cpSlice2,cpSlice)#combining all the images
    # cpSlice = cpSlice2    
    for i in range(len(cpSlice)):
        for j in range(len(cpSlice[0])):
            if maskSlice[i][j] != 255:
                workVal = fillSmallSpace(ogSlice,2,i,j)
                if workVal == 2:
                    cpSlice[i][j] = 255

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cpSlice)

    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    biggestObject = max(sizes)/2
    
    img2 = output.copy()
    img2.fill(0)
    for i in range(0, nb_components):#removal of objects that are smaller than half the size of the biggest detected object
        if sizes[i] >= biggestObject:
            img2[output == i + 1] = 255
    
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            cpSlice[i][j] = img2[i][j]
    maskSlice = cpSlice.copy()
    # imshow("Cp Slice",maskSlice)
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            if maskSlice[i][j] != 255:
                workVal = smoothing(maskSlice,cpSlice,workerRange,i,j)
                if workVal[0] > 3 or workVal[1] > 3 :
                    maskSlice[i][j] = 255
    cpSlice = maskSlice.copy()

    rmSlice = cpSlice.copy()
    drcontours = rmSlice.copy()
    drcontours = cv2.cvtColor(drcontours, cv2.COLOR_GRAY2RGB)
    removeIslands = cv2.pyrDown(rmSlice)
    _, threshed = cv2.threshold(rmSlice, 0, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find maximum contour and draw   
    cmax = max(contours, key = cv2.contourArea) 
    epsilon = 0.002 * cv2.arcLength(cmax, True)
    approx = cv2.approxPolyDP(cmax, epsilon, True)
    cv2.drawContours(drcontours, [approx], -1, (0, 255, 0), 2)
    width, height = rmSlice.shape
    # imshow("Contour", drcontours)
    # waitKey(0)
    #fill maximum contour and draw   
    removeIslands = np.zeros( [width, height, 3],dtype=np.uint8 )
    cv2.fillPoly(removeIslands, pts =[cmax], color=(255,255,255))#removes small islands
    cpSlice = cv2.cvtColor(removeIslands, cv2.COLOR_BGR2GRAY)

    # waitKey(0)
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
                img[i][j] = grey[i][j]#uses the binary mask as a guide to colour the spot colour region

    print("Done with Image: ",numb)
    cv2.imshow("Image", img)
    cv2.imshow("Black image", black_image)
    # cv2.imshow("Black image lines", black_image_lines)

    # cv2.imwrite(data_path+"test_output\\v5\\"+str(numb)+"_thr.jpg",black_image)
    # cv2.imwrite(data_path+"test_output\\v5\\"+str(numb)+"_p.jpg",img)

    # cv2.imwrite(data_path+"train_output\\v5\\"+str(numb)+"_thr.jpg",black_image)
    # cv2.imwrite(data_path+"train_output\\v5\\"+str(numb)+"_p.jpg",img)

    # cv2.imwrite(data_path+"hypothesis\\v5\\"+str(numb)+"_thr.jpg",black_image)
    # cv2.imwrite(data_path+"hypothesis\\v5\\"+str(numb)+"_p.jpg",img)

    cv2.waitKey(0)
            
data_path = os.getcwd()+"\\"

# for i in range(1,11):
#     # if i == 3:
#     #     continue
#     mask = cv2.imread("test_data\\"+str(i)+".jpg")
#     if __name__ == "__main__":
#         p1 = Process(target=run_algorithm,args=[mask,i])
#         p1.start()

# for i in range(1,11):
#     # if i == 3:
#     #     continue
#     mask = cv2.imread("train_data\\"+str(i)+".jpg")
#     if __name__ == "__main__":
#         p1 = Process(target=run_algorithm,args=[mask,i])
#         p1.start()

for i in range(1,8):
    mask = cv2.imread("hypothesis\\"+str(i)+".png")
    if __name__ == "__main__":
        p1 = Process(target=run_algorithm,args=[mask,i])
        p1.start()