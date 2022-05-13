#C1949699
#
#
import math
import random
import re
import time
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


if __name__ == "__main__":
    start_time = time.time() 
    runTestV2()
    print("--- %s seconds ---" % (time.time() - start_time))
    runTimeNumber = str(runTimeCount())   
    saveLog()
    print("Run time number ",runTimeNumber)