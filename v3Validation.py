#C1949699
#
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


interpolationType = [INTER_LINEAR,INTER_CUBIC,INTER_AREA,INTER_BITS,INTER_LANCZOS4,INTER_LINEAR_EXACT]
confidenceList = np.divide(np.subtract(np.arange(15,91,15),5),100)
def selectParameters3():#selecting the parameters that will be swapped for each iteration
    PARAMETERS = list()
    PARAMETERS.clear()
    PARAMETERS.append(random.choice(confidenceList))
    PARAMETERS.append(random.choice(interpolationType))
    return PARAMETERS

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
    numRange = 2
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
        # listOfEvaluations = []
        # evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            evaluationParameters = manager.list()
            listOfEvaluations = manager.list()
            PARAMETERS = manager.list()
            jobs = []
            # listOfBinaryMask.fromkeys(datasetTrain)
            for i in range(4):
                # findOptimalParams(listOfEvaluations,evaluationParameters,dictOfBinaryMask)
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            print(listOfEvaluations)
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



def findOptimalParams(listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain):
    for i in range(1):
        PARAMETERS = selectParameters3()
        print("PARAMS ",PARAMETERS)
        if PARAMETERS in evaluationParameters:
            break
        evaluationParameters.append(PARAMETERS)#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
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


if __name__ == "__main__":
    start_time = time.time() 
    runTestV3()
    print("--- %s seconds ---" % (time.time() - start_time))
    runTimeNumber = str(runTimeCount())   
    saveLog()
    print("Run time number ",runTimeNumber)
