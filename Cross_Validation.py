#C1949699
#
#Joseph Wilson-Hall
import math
import random
import re
import time
from turtle import numinput
from cv2 import INTER_AREA, INTER_BITS, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_LINEAR_EXACT, INTER_MAX, imread, imshow, waitKey
import numpy as np
import cv2
import os
from multiprocessing import Process, Manager

data_path = os.getcwd()
dataset_path = data_path+"\\cross_validation\\"
imageDataLocation = dataset_path+"\\images\\"
truthDataLocation = dataset_path+"\\truth\\"
LOG = []
datasetTrain = []
datasetVal = []
PARAMETERS = []
runTimeNumber = int(1)

def calculateSTD(evaluationList, evaluationMean):
    n = 1
    if len(evaluationList)!=1:
        n = len(evaluationList)-1
    sumX = 0
    for score in evaluationList:
        sumX+=(score-evaluationMean)**2
    standardDeviation = sumX / n
    standardDeviation = math.sqrt(standardDeviation)
    return standardDeviation
    

def runTimeCount():
    i = 1
    fileList = os.listdir(data_path)
    while "log"+str(i)+".txt" in fileList:
            i+=1    
    return int(i)

def saveLog():    
    i = 1
    fileList = os.listdir(data_path)
    if "log1.txt" not in fileList:
        with open("log"+str(i)+".txt","a") as f:
            for line in LOG:
                f.write(line+"\n")
    else:
        while "log"+str(i)+".txt" in fileList:
            i+=1
        with open("log"+str(i)+".txt","a") as f:
            for line in LOG:
                f.write(line+"\n")
    LOG.clear()

def addToLog(line,varname):
    # print("Line",line)
    if varname == "BinaryMasks":
        LOG.append(varname,line)
    elif isinstance(line, list):
        # log.append(varname)
        LOG.append(str(varname+" "+f'{line}'.split('=')[0]))
    elif isinstance(line, str or int):
        # log.append(varname)
        LOG.append(str(varname)+" "+str(line))
    elif isinstance(line, float):
        LOG.append(str(varname)+" "+str(line))

def calc_IoU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    # print(mask1_area, " : ", mask2_area)
    intersection = np.count_nonzero(np.logical_and( mask1,  mask2))
    # print("intersection",intersection)
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou




# def runTest():
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):   
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def select_new_dataset():
    datasetTrain.clear()
    datasetVal.clear()
    files = os.listdir(imageDataLocation)
    for i in range(14):
        file = random.choice(files) 
        while file in datasetTrain:
            file = random.choice(files)
        datasetTrain.append(file)
    for file in files:
        if file not in datasetTrain:
            datasetVal.append(file)
    # datasetTrain.sort(key=natural_keys)
    # datasetVal.sort(key=natural_keys)

#Region v1
#parameters that will be adjusted include: 
# - thresholding method
# - saliency type
# -   
#thresholding types need to be defined here
listOfMethods = [cv2.THRESH_OTSU,cv2.THRESH_TRIANGLE]
listOfTypes = [cv2.THRESH_BINARY,cv2.THRESH_TOZERO]
# saliencyType = [int(1),int(2)]
#[cv2.saliency.StaticSaliencyFineGrained_create(),cv2.saliency.StaticSaliencySpectralResidual_create()]
def selectParameters():#selecting the parameters that will be swapped for each iteration
    PARAMETERS.clear()
    PARAMETERS.append(random.choice(listOfMethods))
    PARAMETERS.append(random.choice(listOfTypes))
    # PARAMETERS.append(random.choice(saliencyType))

def v1(img,numb,PARAMETERS,dictOfBinaryMask):
    # if int(PARAMETERS[2]) == 1:
    #     saliency = cv2.saliency.StaticSaliencyFineGrained_create()# calculating the static saliency fine grained map
    # elif int(PARAMETERS[2]) == 2:
    #     saliency = cv2.saliency.StaticSaliencySpectralResidual_create()# calculating the static saliency residual map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()# calculating the static saliency fine grained map
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255#opencv returns a floating point number when computing saliency so multiplying by 255 sets it to be within the limits of 0-255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, PARAMETERS[0] | PARAMETERS[1])[1]#thresholding saliency map using otsu's algorithm to get a binary mask that is used when choosing areas that should be in colour
    threshMap = cv2.threshold(threshMap.astype("uint8"), 80, 255, cv2.THRESH_BINARY)[1]

    dictOfBinaryMask[numb] = threshMap

def runTestV1():
    listTestEvaluations = []
    for i in range(50):
        if i % 10 == 0:
            print(i*2,"% Done")
        addToLog(i,"Loop: ")
        select_new_dataset()
        addToLog(datasetTrain,f'{datasetTrain=}'.split('=')[0])
        addToLog(datasetVal,f'{datasetVal=}'.split('=')[0])
        # listOfBinaryMask = []
        # for i in range(10):
        listOfEvaluations = []
        evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            # listOfBinaryMask.fromkeys(datasetTrain)
            for i in range(100):#Only runs for the maximum number of combinations
                selectParameters()
                if PARAMETERS in evaluationParameters:
                    continue
                evaluationParameters.append(PARAMETERS.copy())#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
                # print(evaluationParameters)
                # addToLog(PARAMETERS.copy(),"Parameters")
                jobs = []
                for image in datasetTrain:
                    current = imread(imageDataLocation+image)
                    p1 = Process(target=v1,args=[current,image,PARAMETERS,dictOfBinaryMask])
                    p1.start()
                    jobs.append(p1)                                    
                for job in jobs:
                    job.join()
                averageIOU=0
                for image in datasetTrain:
                    imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                    max_IoU = 0
                    for i in range (1,6):
                        mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                        mask2 = imread(truthDataLocation+mask2)
                        mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                        max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                    averageIOU+=max_IoU
                averageIOU /= len(datasetTrain)
                # print(averageIOU)
                listOfEvaluations.append(averageIOU)
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            # dictOfBinaryMask.clear()
            jobs.clear()
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v1,args=[current,image,optimalParams,dictOfBinaryMask])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            averageIOU=0
            for image in datasetVal:
                imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                max_IoU = 0
                for i in range (1,6):
                    mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                    mask2 = imread(truthDataLocation+mask2)
                    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                    max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                averageIOU+=max_IoU
            averageIOU /= len(datasetVal)
            addToLog(optimalParams,"Optimal Parameters")
            addToLog(averageIOU,"Evaluation Score Validation Set")
            listTestEvaluations.append(averageIOU)
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("V1","Pipeline: ")

#EndRegion
#Region v2
#parameters that will be adjusted include: 
# - confidence threshold value
# - saliency type
# - image thresholding method
listOfMethods = [cv2.THRESH_OTSU,cv2.THRESH_TRIANGLE]
listOfTypes = [cv2.THRESH_BINARY,cv2.THRESH_TOZERO]
confidenceList = np.divide(np.arange(10,91,15),100)
def selectParameters2():#selecting the parameters that will be swapped for each iteration
    PARAMETERS.clear()
    PARAMETERS.append(random.choice(listOfMethods))
    PARAMETERS.append(random.choice(listOfTypes))
    PARAMETERS.append(random.choice(confidenceList))

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net2 = cv2.dnn_DetectionModel(weightsPath,configPath)
net2.setInputSize(320,320)
net2.setInputScale(1.0/127.5)
net2.setInputMean((127.5,127.5,127.5))
net2.setInputSwapRB(True)

def v2(img,numb,PARAMETERS,dictOfBinaryMask):
    thres = PARAMETERS[2]
    
    classIds, confs, bbox = net2.detect(img,confThreshold = thres)
    boxes = []
    confidenceList = []
    if not isinstance(classIds,tuple):
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
    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, PARAMETERS[0] | PARAMETERS[1])[1]#globally thresholding the image using otsu's algorithm creating a binary mask
    threshMap = cv2.threshold(threshMap.astype("uint8"), 80, 255, cv2.THRESH_BINARY)[1]
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
    # print(minX,":",minY,"  ",maxX,":",maxY)
    if bool(boxes):#if the list is empty it will return the threshmap
        dictOfBinaryMask[numb] = threshMap
    else:
        for b in boxes: # finding the bounding boxes area
            x1 = b[0]
            y1 = b[1]
            x2 = b[2]+b[0]#the b[2] and b[3] represents a length not a coordinate so adding b[0] to b[2] finds the coord of the second point
            y2 = b[3]+b[1]
            
            for i in range(y1, y2):
                for j in range(x1, x2):
                    if (threshMap[i][j] == 255):#selecting pixels that are only inside the bounding boxes and where the mask is white, making the ROI in colour
                        newThrMap[i][j] = 255

        dictOfBinaryMask[numb] = newThrMap

def runTestV2():
    listTestEvaluations = []
    for i in range(50):
        if i % 10 == 0:
            print(i*2,"% Done")
        addToLog(i,"Loop: ")
        select_new_dataset()
        addToLog(datasetTrain,f'{datasetTrain=}'.split('=')[0])
        addToLog(datasetVal,f'{datasetVal=}'.split('=')[0])
        # listOfBinaryMask = []
        # for i in range(10):
        listOfEvaluations = []
        evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            # listOfBinaryMask.fromkeys(datasetTrain)
            for i in range(1000):#Only runs for the maximum number of combinations
                selectParameters2()
                if PARAMETERS in evaluationParameters:
                    continue
                evaluationParameters.append(PARAMETERS.copy())#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
                # print(evaluationParameters)
                # addToLog(PARAMETERS.copy(),"Parameters")
                jobs = []
                for image in datasetTrain:
                    current = imread(imageDataLocation+image)
                    p1 = Process(target=v2,args=[current,image,PARAMETERS,dictOfBinaryMask])
                    p1.start()
                    jobs.append(p1)                    
                for job in jobs:
                    job.join()
                averageIOU=0
                for image in datasetTrain:
                    imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                    max_IoU = 0
                    for i in range (1,6):
                        mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                        mask2 = imread(truthDataLocation+mask2)
                        mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                        max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                    averageIOU+=max_IoU                    
                averageIOU /= len(datasetTrain)
                # print(averageIOU)
                listOfEvaluations.append(averageIOU)
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            jobs.clear()
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v2,args=[current,image,optimalParams,dictOfBinaryMask])
                p1.start()
                jobs.append(p1)                
            for job in jobs:
                job.join()
            averageIOU=0
            for image in datasetVal:
                imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                max_IoU = 0
                for i in range (1,6):
                    mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                    mask2 = imread(truthDataLocation+mask2)
                    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                    max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                averageIOU+=max_IoU
            averageIOU /= len(datasetVal)
            addToLog(optimalParams,"Optimal Parameters")
            addToLog(averageIOU,"Evaluation Score Validation Set")
            listTestEvaluations.append(averageIOU)
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("V2","Pipeline: ")



#EndRegion
#Region v3
#parameters that will be adjusted include: 
# - confidence threshold value
# - interpolation type
# - 
interpolationType = [INTER_LINEAR,INTER_CUBIC,INTER_AREA,INTER_BITS,INTER_LANCZOS4,INTER_LINEAR_EXACT]
confidenceList = np.divide(np.subtract(np.arange(15,91,15),5),100)
def selectParameters3():#selecting the parameters that will be swapped for each iteration
    PARAMETERS.clear()
    PARAMETERS.append(random.choice(confidenceList))
    PARAMETERS.append(random.choice(interpolationType))

net3 = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
def v3(img,numb,PARAMETERS,dictOfBinaryMask):
    thres = PARAMETERS[0]
    
    H, W, _ = img.shape
    # Create black image with same dimensions as input image
    black_image = np.zeros((H, W), np.uint8)

    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net3.setInput(blob)

    boxes, masks = net3.forward(["detection_out_final", "detection_masks"])
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
        mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=PARAMETERS[1])
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Get mask coordinates
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], 255)

    dictOfBinaryMask[numb] = black_image

   
def runTestV3():
    listTestEvaluations = []
    numRange = 50
    for i in range(numRange):
        if i % 10 == 0:
            print(i*2,"% Done")
        elif numRange < 10:
            print(i,"iteration")
        addToLog(i,"Loop: ")
        select_new_dataset()
        addToLog(datasetTrain,f'{datasetTrain=}'.split('=')[0])
        addToLog(datasetVal,f'{datasetVal=}'.split('=')[0])
        # listOfBinaryMask = []
        # for i in range(10):
        listOfEvaluations = []
        evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            # listOfBinaryMask.fromkeys(datasetTrain)
            for i in range(1000):#Only runs for the maximum number of combinations
                selectParameters3()
                if PARAMETERS in evaluationParameters:
                    continue
                evaluationParameters.append(PARAMETERS.copy())#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
                # print(evaluationParameters)
                # addToLog(PARAMETERS.copy(),"Parameters")
                jobs = []
                for image in datasetTrain:
                    current = imread(imageDataLocation+image)
                    p1 = Process(target=v3,args=[current,image,PARAMETERS,dictOfBinaryMask])
                    p1.start()
                    jobs.append(p1)                    
                for job in jobs:
                    job.join()
                averageIOU=0
                for image in datasetTrain:
                    imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                    max_IoU = 0
                    for i in range (1,6):
                        mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                        mask2 = imread(truthDataLocation+mask2)
                        mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                        max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                    averageIOU+=max_IoU                    
                averageIOU /= len(datasetTrain)
                # print(averageIOU)
                listOfEvaluations.append(averageIOU)
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            jobs.clear()
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v3,args=[current,image,optimalParams,dictOfBinaryMask])
                p1.start()
                jobs.append(p1)                
            for job in jobs:
                job.join()
            averageIOU=0
            for image in datasetVal:
                imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                max_IoU = 0
                for i in range (1,6):
                    mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                    mask2 = imread(truthDataLocation+mask2)
                    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                    max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                averageIOU+=max_IoU
            averageIOU /= len(datasetVal)
            addToLog(optimalParams,"Optimal Parameters")
            addToLog(averageIOU,"Evaluation Score Validation Set")
            listTestEvaluations.append(averageIOU)
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("V3","Pipeline: ")



#EndRegion
#Region v4
#parameters that will be adjusted include: 
# - confidence threshold value
# - range of workers
# - 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# test_data_path = os.path.dirname(os.getcwd())+"\\data\\new_test_data\\"
smallRangeWorker = np.arange(55,101,5)
LongRangeWorker = np.arange(3,31,3)
confidenceList = np.divide(np.subtract(np.arange(15,85,15),5),100)
def selectParameters4():#selecting the parameters that will be swapped for each iteration
    PARAMETERS = list()
    PARAMETERS.append(random.choice(confidenceList))
    PARAMETERS.append(random.choice(smallRangeWorker))
    PARAMETERS.append(random.choice(LongRangeWorker))
    return PARAMETERS

def workFunc(img, workerRange,x,y):
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
    # return up,down,left,right
def fillSmallSpace(img, workerRange,x,y):
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
    # return up,down,left,right

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
    return temp


def v4(img,numb,PARAMETERS,dictOfBinaryMask):
    thres = PARAMETERS[0]
    classIds, confs, bbox = net.detect(img,confThreshold = thres)
    # print("BOX: ",bbox[0])
    # x1 = bbox[0][0]
    # y1 = bbox[0][1]
    # x2 = bbox[0][2]
    # y2 = bbox[0][3]
    boxes = []
    confidenceList = []
    imgNew = img.copy()
    imgNew.fill(0)
    if not isinstance(classIds,tuple):
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            boxes.append([x1,y1,x2,y2])
            confidenceList.append(confs)
            # cv2.rectangle(img,[x1,y1,x2,y2],color=(0,0,255),thickness=2)

        maxX = 0
        maxY = 0
        minX = len(img)*len(img[0])
        minY = len(img)*len(img[0])
        for b in boxes:
            for i in range(b[0],b[2]+b[0]):
                for j in range(b[1],b[3]+b[1]):
                    imgNew[j][i] = img[j][i]
        
        for b in boxes:
            maxX = max(maxX,b[0]+b[2])
            maxY = max(maxY,b[1]+b[3])
            minX = min(minX,b[0])
            minY = min(minY,b[1])
            
        
        fMask = np.copy(img)
        fMask.fill(0)
        ogSlice = img[minY:maxY,minX:maxX]
        ogSlice = find_optimal_lines(ogSlice,numb)
        ogSlice = cv2.dilate(ogSlice,(7,7))
        
        newSlice = ogSlice.copy()
        newSlice.fill(0)

        for b in boxes:
            for i in range(b[0]-minX,(b[2]+b[0])-minX):
                for j in range(b[1]-minY,(b[3]+b[1])-minY):
                    newSlice[j][i] = ogSlice[j][i]

        ogSlice = newSlice
        cpSlice = ogSlice.copy()
        
        smallRange = int(max(len(ogSlice)/PARAMETERS[1],len(ogSlice[0])/PARAMETERS[1]))
        longRange = int(max(len(ogSlice)/PARAMETERS[2],len(ogSlice[0])/PARAMETERS[2]))
        workVal = 0
        
        
        
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = fillSmallSpace(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 2:
                        cpSlice[i][j] = 255            
        
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = workFunc(ogSlice,longRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal > 3:
                        cpSlice[i][j] = 255
        ogSlice = cpSlice.copy()
        # print(tempRange)
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = fillSmallSpace(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 2:
                        cpSlice[i][j] = 255
                        ogSlice[i][j] = 255
        # imshow("ogSlice",ogSlice)
        # imshow("cpSlice",cpSlice)
        # waitKey(0)
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = workFunc(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 4:
                        cpSlice[i][j] = 255
                elif(ogSlice[i][j] != 0):
                    if workVal < 2:
                        # ogSlice[i][j] = 0
                        cpSlice[i][j] = 0
        
        # imshow("ogSlice",ogSlice)
        # imshow("cpSlice",cpSlice)
        # waitKey(0)    
        rmSlice = cpSlice.copy()
        drcontours = rmSlice.copy()
        drcontours = cv2.cvtColor(drcontours, cv2.COLOR_GRAY2RGB)
        removeIslands = cv2.pyrDown(rmSlice)
        _, threshed = cv2.threshold(rmSlice, 0, 255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("Cont",contours)
        #find maximum contour and draw
        
        # print("NUMB",numb)
        if len(contours) > 0:
            cmax = max(contours, key = cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(cmax, True)
            approx = cv2.approxPolyDP(cmax, epsilon, True)
            cv2.drawContours(drcontours, [approx], -1, (0, 255, 0), 2)
            width, height = rmSlice.shape
            # imshow("Contour", drcontours)
            # waitKey(0)
            #fill maximum contour and draw   
            removeIslands = np.zeros( [width, height, 3],dtype=np.uint8 )
            cv2.fillPoly(removeIslands, pts =[cmax], color=(255,255,255))
            cpSlice = cv2.cvtColor(removeIslands, cv2.COLOR_BGR2GRAY)
            
        # print("Cont",cmax)
        

        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                fMask[minY+i][minX+j] = cpSlice[i][j]
        fMask = cv2.cvtColor(fMask, cv2.COLOR_BGR2GRAY)
        dictOfBinaryMask[numb] = fMask
    else:
        fMask = img.copy()
        fMask = cv2.cvtColor(fMask, cv2.COLOR_BGR2GRAY)
        fMask.fill(0)
        dictOfBinaryMask[numb] = fMask

def findOptimalParams(listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain):
    for i in range(1):
        PARAMETERS = selectParameters4()
        print("PARAMS ",PARAMETERS)
        if PARAMETERS in evaluationParameters:
            break
        evaluationParameters.append(PARAMETERS)#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
        # print(evaluationParameters)
        # addToLog(PARAMETERS.copy(),"Parameters")
        jobs = []
        for image in datasetTrain:
            current = imread(imageDataLocation+image)
            p1 = Process(target=v4,args=[current,image,PARAMETERS,dictOfBinaryMask])
            p1.start()
            jobs.append(p1)                    
        for job in jobs:
            job.join()
        averageIOU=0
        for image in datasetTrain:
            imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
            max_IoU = 0
            for i in range (1,6):
                mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                mask2 = imread(truthDataLocation+mask2)
                mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
            averageIOU+=max_IoU                    
        averageIOU /= len(datasetTrain)
        # print(averageIOU)
        listOfEvaluations.append(averageIOU)


def runTestV4():
    listTestEvaluations = []
    numRange = 50
    for i in range(numRange):
        if i % 10 == 0 and numRange > 9:
            print(i*2,"% Done")
        elif numRange < 10:
            print(i,"iteration")
        addToLog(i,"Loop: ")
        select_new_dataset()
        addToLog(datasetTrain,f'{datasetTrain=}'.split('=')[0])
        addToLog(datasetVal,f'{datasetVal=}'.split('=')[0])
        # listOfBinaryMask = []
        # for i in range(10):
        listOfEvaluations = []
        evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            evaluationParameters = manager.list()
            listOfEvaluations = manager.list()
            PARAMETERS = manager.list()
            jobs = []
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            print("Length of listOfEvaluations",len(listOfEvaluations))
            print("Length of evaluationParameters",len(evaluationParameters))
            print("Length of listTestEvaluations",len(listTestEvaluations))
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            print("Second part")
            
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v4,args=[current,image,optimalParams,dictOfBinaryMask])
                p1.start()
                jobs.append(p1)                
            for job in jobs:
                job.join()
            averageIOU=0
            for image in datasetVal:
                imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                max_IoU = 0
                for i in range (1,6):
                    mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                    mask2 = imread(truthDataLocation+mask2)
                    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                    max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                averageIOU+=max_IoU
            averageIOU /= len(datasetVal)
            addToLog(optimalParams,"Optimal Parameters")
            addToLog(averageIOU,"Evaluation Score Validation Set")
            listTestEvaluations.append(averageIOU)
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("V4","Pipeline: ")

#EndRegion
#Region v5
def rV5(img,numb):
    confsThr = 0.4
    boxThr = 0.5
    #Loading in rcnn mask model
    net5 = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph_coco.pb","dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    #Loading in bounding box model
    net5 = cv2.dnn_DetectionModel("frozen_inference_graph.pb","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    net5.setInputSize(320,320)
    net5.setInputScale(1.0/127.5)
    net5.setInputMean((127.5,127.5,127.5))
    net5.setInputSwapRB(True)
    #load our input image and grab its spatial dimensions

    def find_optimal_lines_v5(ogSlice,numb):
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

    def fillSmallSpace_v5(img, workerRange,x,y):
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

    def smoothing(mask,lineImage, workerRange,x,y):
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

    def v5(img, numb):
        
        H, W, _ = img.shape
        # Create black image with same dimensions as input image
        black_image = np.zeros((H, W), np.uint8)
        # Detect objects inside input image
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net5.setInput(blob)
        boxes, masks = net5.forward(["detection_out_final", "detection_masks"])
        _, _, boxes2 = net5.detect(img,confThreshold = boxThr)
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
        for box in boundingBox:
            maxX = max(maxX,box[2])
            maxY = max(maxY,box[3])
            minX = min(minX,box[0])
            minY = min(minY,box[1])

        ogSlice = img[minY:maxY, minX:maxX]
        maskSlice = black_image[minY:maxY, minX:maxX]
        ogSlice = find_optimal_lines_v5(ogSlice,numb)
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
                    if workVal[0] >= 2 and workVal[1] >= 2 :
                        cpSlice[i][j] = 255
                    # elif workVal[0] > 3:
                    #     cpSlice[i][j] = 255
        # ogSlice = cpSlice.copy()
        cpSlice2 = cpSlice.copy()
        cpSlice = cv2.bitwise_or(maskSlice,ogSlice)
        cpSlice = cv2.bitwise_or(cpSlice2,cpSlice)
        # cpSlice = cpSlice2
        smallWorkerRange = int(max(len(ogSlice)/100,len(ogSlice[0])/100))
        for i in range(len(cpSlice)):
            for j in range(len(cpSlice[0])):
                if maskSlice[i][j] != 255:
                    workVal = fillSmallSpace_v5(ogSlice,2,i,j)
                    if workVal == 2:
                        cpSlice[i][j] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cpSlice)
        # imshow_components(output)

        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        biggestObject = max(sizes)/2
        
        img2 = output.copy()
        img2.fill(0)
        for i in range(0, nb_components):
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
        cv2.fillPoly(removeIslands, pts =[cmax], color=(255,255,255))
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
                    img[i][j] = grey[i][j]

        print("Done with Image: ",numb)
        # cv2.imshow("Image", img)
        # cv2.imshow("Black image", black_image)
        # cv2.imshow("Black image lines", black_image_lines)

        # cv2.imwrite(data_path+"test_output\\v5\\"+str(numb)+"_thr.jpg",black_image)
        # cv2.imwrite(data_path+"test_output\\v5\\"+str(numb)+"_p.jpg",img)

        cv2.imwrite(data_path+"train_output\\v5\\"+str(numb)+"_thr.jpg",black_image)
        cv2.imwrite(data_path+"train_output\\v5\\"+str(numb)+"_p.jpg",img)

        # cv2.waitKey(0)
#EndRegion


if __name__ == "__main__":
    runTimeNumber = str(runTimeCount())   
    start_time = time.time() 
    runTestV2()
    # testimg = imread(imageDataLocation+"5.jpg")
    # tempDict = dict()
    # v2(testimg,"5.jpb",[cv2.THRESH_OTSU,cv2.THRESH_BINARY,0.5],tempDict)
    print("--- %s seconds ---" % (time.time() - start_time))
    saveLog()
    print("Run time number ",runTimeNumber)




