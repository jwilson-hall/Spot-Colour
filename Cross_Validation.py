#C1949699
#
#
#
#
#
import random
import numpy as np
import cv2
import os
from multiprocessing import Process

data_path = os.getcwd()+"\\cross_validation\\"

log = []
datasetTrain = []
datasetVal = []
def saveLog():
    
    i = 1
    fileList = os.listdir(data_path)
    if "log1.txt" not in fileList:
        with open("log"+str(i)+".txt","a") as f:
            for line in log:
                f.write(line+"\n")
    else:
        while "log"+str(i)+".txt" in fileList:
            i+=1
        with open("log"+str(i)+".txt","a") as f:
            for line in log:
                f.write(line+"\n")
    print(log)
    log.clear()

def addToLog(line):
    if isinstance(line, list):
        log.append(f'{line}'.split('=')[0])
    if isinstance(line, str or int):
        log.append(str(line))

def select_new_dataset():
    dataLocation = data_path+"\\images\\"
    datasetTrain.clear()
    datasetVal.clear()
    files = os.listdir(dataLocation)
    for i in range(14):
        file = random.choice(files) 
        while file in datasetTrain:
            file = random.choice(files)
        datasetTrain.append(file)
    for file in files:
        if file not in datasetTrain:
            datasetVal.append(file)


if __name__ == "__main__":
    select_new_dataset()
    addToLog(datasetTrain)
    addToLog(datasetVal)
    saveLog()


#Region v1
#parameters that will be adjusted include: 
# - 
# - 
# -   
def v1(img, numb):
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()# calculating the static saliency fine grained map
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255#opencv returns a floating point number when computing saliency so multiplying by 255 sets it to be within the limits of 0-255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#thresholding saliency map using otsu's algorithm to get a binary mask that is used when choosing areas that should be in colour
    
    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #creating a greyscale copy of the original image
    output_img = cv2.cvtColor(output_img,cv2.COLOR_GRAY2RGB)#converting the greyscale image back to rgb values so that we can set the greyscale value to the RGB tuple
    
    
    for i in range(len(threshMap)):
        for j in range(len(threshMap[0])):
            if (threshMap[i][j]==0):
                img[i][j] = output_img[i][j]

    # cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_thr.jpg",threshMap)
    # cv2.imwrite(data_path+"test_output\\v1\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"test_output\\v1\\"+str(numb)+"_thr.jpg",threshMap)
    cv2.imshow("Output image",img)
    cv2.imshow("Output", threshMap)
    cv2.waitKey(0)
#EndRegion
#Region v2
#parameters that will be adjusted include: 
# - confidence threshold value
# - saliency type
# - image thresholding method
def v2(img, numb):
    thres = 0.45 # Threshold to detect object

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img,confThreshold = thres)
    # print("BOX: ",bbox[0])
    # x1 = bbox[0][0]
    # y1 = bbox[0][1]
    # x2 = bbox[0][2]
    # y2 = bbox[0][3]
    boxes = []
    confidenceList = []
    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        boxes.append([x1,y1,x2,y2])
        confidenceList.append(confidence)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#globally thresholding the image using otsu's algorithm creating a binary mask

    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#creating a greyscale copy of the original image
    testGrey = img.copy()#creates a colour copy of the original image
    for i in range(len(img)):
        for j in range(len(img[0])):
            testGrey[i][j] = np.array([output_img[i][j],output_img[i][j],output_img[i][j]])
    
    maxX = 0
    maxY = 0
    minX = len(img)*len(img[0])
    minY = len(img)*len(img[0])
    newThrMap = threshMap.copy()
    newThrMap.fill(0)
    # boxes.remove(boxes[0])
    for b in boxes:
        maxX = max(maxX,b[0],b[2])#finding the overall bounding area of all the boxes
        maxY = max(maxY,b[1],b[3])
        minX = min(minX,b[2],b[0])
        minY = min(minY,b[3],b[1])
    print(minX,":",minY,"  ",maxX,":",maxY)
    for b in boxes: # finding the bounding boxes area
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]+b[0]#the b[2] and b[3] represents a length not a coordinate so adding b[0] to b[2] finds the coord of the second point
        y2 = b[3]+b[1]
        
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (threshMap[i][j] == 255 and (j >= x1 and i >= y1) and (j <= x2 and i <= y2)):#selecting pixels that are only inside the bounding boxes and where the mask is white, making the ROI in colour
                    testGrey[i][j] = img[i][j]
                    newThrMap[i][j] = 255

    # cv2.imshow("Output",testGrey)
    # cv2.imshow("Threshold map",newThrMap)
    # data_path = os.path.dirname(os.getcwd())+"\\data\\"
    # Writing Images
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_p.jpg",testGrey)#writing output mask and final product of the use of that mask
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_thr.jpg",newThrMap)
    # cv2.imwrite(data_path+"test_output\\v2\\"+str(numb)+"_p.jpg",testGrey)#writing output mask and final product of the use of that mask
    # cv2.imwrite(data_path+"test_output\\v2\\"+str(numb)+"_thr.jpg",newThrMap)
    # cv2.imshow("Output image",output_img)
    # cv2.imshow("Output", saliencyMap)
    # cv2.waitKey(0)
#EndRegion
#Region v3
#parameters that will be adjusted include: 
# - confidence threshold value
# - 
# - 
def v3(img, numb):
    thres = 0.5
    net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    H, W, _ = img.shape
	# Create black image with same dimensions as input image
    black_image = np.zeros((H, W, 1), np.uint8)

    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        confs = box[2]
        if confs < thres:
            continue

        # Get box coordinates
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (x1, y1, x2, y2) = box.astype('int')
        roi_Width = x2 - x1
        roi_Height = y2 - y1
        roi = black_image[y1: y2, x1: x2]

        #creating mask size based on the region of interest bounding box
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_Width, roi_Height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Get mask coordinates
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], 255)

        cv2.imshow("roi", roi)
        cv2.waitKey(0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if black_image[i][j] != 255:			
                img[i][j] = grey[i][j]


    cv2.imshow("Image", img)
    cv2.imshow("Black image", black_image)
    cv2.waitKey(0)
    print("Done with image ",numb)
    # cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
    # cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
#EndRegion
#Region v4

#EndRegion
#Region v5

#EndRegion